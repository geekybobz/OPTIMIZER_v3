"""Durable service transactions, named experiments and numerical ancestry."""
from __future__ import annotations
import copy
import inspect
from pathlib import Path
from uuid import uuid4
from optimizer.checkpoints.codec import _Codec, _hash, _json
from optimizer.persistence.store import _Store, _environment
from optimizer.operations.contracts import ObjectRef, OperationRequest, OperationResponse, _ServiceError
from optimizer.session import Session, _Stored, _owned, _physical, _compatible, _size
from optimizer.state import RunState
from optimizer.result import OptimizerResult
from optimizer.system_olss import ShootingResult

_CERTIFIED={'optimizers.adam','optimizers.momentum','shooting.solve_shooting'}

def _clone(session):
    staged=Session(max_bytes=session.max_bytes,max_objects=session.max_objects)
    staged._contexts={n:copy.copy(c) for n,c in session._contexts.items()}
    staged._objects={n:copy.copy(v) for n,v in session._objects.items()}
    staged._bindings=dict(session._bindings)
    staged._pruned=set(session._pruned)
    staged._selected,staged._revision=session._selected,session._revision
    return staged

def _adopt(target,staged):
    target._contexts,target._objects,target._bindings=staged._contexts,staged._objects,staged._bindings
    target._selected,target._revision=staged._selected,staged._revision
    target._pruned=staged._pruned

class _Persistence:
    def __init__(self,service):
        self.service=service; self.store=None
        self.methods={'project.init':self.init,'project.open':self.open,'project.close':self.close,
            'project.fork':self.fork,'project.list':self.list_projects,'project.status':self.status,
            'session.save':self.save,'session.restore':self.restore,'session.undo':self.undo,'session.redo':self.redo,
            'checkpoints.save':self.save_object,'checkpoints.load':self.load,'checkpoints.list':self.list_checkpoints,
            'checkpoints.pin':self.pin,'checkpoints.unpin':self.unpin,
            'storage.status':self.status,'storage.configure':self.configure,'storage.plan':self.plan,'storage.apply':self.apply}
        self.guard=None
        self.staged=None; self.index=None; self.files={}; self.changed=False; self.track=True

    def descriptors(self):
        return [{'id':name,'name':name.split('.')[-1],'classification':'lifecycle','effect':'session',
                 'signature':str(inspect.signature(method)),
                 'reason':'Shared service lifecycle adapter; initialize/open a project before checkpoint/storage operations.'}
                for name,method in self.methods.items()]

    def require(self):
        if self.store is None: raise _ServiceError('missing_project','Initialize or open an experiment first.')
        return self.store

    def _recipe_files(self,recipe):
        if not recipe.get('path'): raise _ServiceError('unsupported_persistence','Durable systems require an explicit source file.')
        files={}; sources=[(recipe['path'],recipe['sha256'])]
        declaration=recipe.get('checkpoint',{})
        if set(declaration)-{'stateless','dependencies'} or type(declaration.get('stateless',False)) is not bool:
            raise _ServiceError('invalid_arguments','checkpoint accepts stateless:bool and dependencies:list[path].')
        dependencies=declaration.get('dependencies',[])
        if not isinstance(dependencies,list) or any(not isinstance(p,str) for p in dependencies):
            raise _ServiceError('invalid_arguments','Checkpoint dependencies must be explicit file paths.')
        captured=recipe.get('dependency_hashes')
        for value in dependencies:
            path=Path(value).expanduser()
            if not path.is_absolute(): path=Path(recipe['path']).parent/path
            path=path.resolve(strict=True)
            if path.stat().st_size>self.service.session.max_bytes:
                raise _ServiceError('codec_limit','Dependency exceeds foreground read budget.')
            data=path.read_bytes(); digest=_hash(data)
            if captured is not None and captured.get(str(path))!=digest:
                raise _ServiceError('source_changed','Declared input/dependency changed.')
            sources.append((str(path),digest))
        for value,digest in sources:
            path=Path(value)
            if path.stat().st_size>self.service.session.max_bytes:
                raise _ServiceError('codec_limit','Source/input exceeds foreground read budget.')
            data=path.read_bytes()
            if _hash(data)!=digest: raise _ServiceError('source_changed','Project source changed; exact state is incompatible.')
            files[f'inputs/{digest}/payload']=data
        if sum(len(data) for data in files.values())>self.service.session.max_bytes:
            raise _ServiceError('codec_limit','Source bundle exceeds foreground read budget.')
        if declaration.get('stateless'):
            import ast
            declared={Path(path).resolve() for path,digest in sources}
            for source in declared:
                if source.suffix!='.py': continue
                tree=ast.parse(source.read_text())
                for node in ast.walk(tree):
                    names=[alias.name for alias in node.names] if isinstance(node,ast.Import) else ([node.module] if node.module else [a.name for a in node.names]) if isinstance(node,ast.ImportFrom) else []
                    for module in names:
                        base=source.parent
                        if isinstance(node,ast.ImportFrom):
                            for _ in range(max(0,node.level-1)): base=base.parent
                        candidate=base.joinpath(*module.split('.'))
                        for local in (candidate.with_suffix('.py'),candidate/'__init__.py'):
                            if local.is_file() and local.resolve() not in declared:
                                raise _ServiceError('unsupported_persistence',f'Undeclared local dependency: {local}.')
        recipe['dependency_hashes']={path:digest for path,digest in sources[1:]}
        return files

    def _level(self,stored):
        physical=_physical(stored.value)
        if physical is None: return 'analysis_only'
        key=stored.method_key
        declaration=(stored.recipe or {}).get('checkpoint',{})
        if key is not None and key[0] in _CERTIFIED and declaration.get('stateless') is True and 'dependencies' in declaration:
            if 'ephemeral_callable' not in repr(key):
                if key[0] in {'optimizers.adam','optimizers.momentum'} and dict(key[1]).get('variant')!=({'optimizers.adam':'adam','optimizers.momentum':'heavy_ball'}[key[0]]):
                    return 'controls_only'
                if key[0]=='shooting.solve_shooting' and dict(key[1]).get('method')!='lm':
                    return 'variables_only'
                return 'exact'
        return 'controls_only' if hasattr(physical,'as_matrix') else 'variables_only'

    def _pack(self,oid,stored):
        codec=_Codec(max_bytes=self.service.session.max_bytes)
        recipe=_owned(stored.recipe)
        if recipe: self.files.update(self._recipe_files(recipe))
        stored.recipe=recipe
        level=self._level(stored)
        value=stored.value
        payload={'value':value,'context':stored.context,'configuration':stored.configuration,
                 'method_key':stored.method_key if level=='exact' else None,'recipe':recipe,'resume_level':level,
                 'environment':_environment(),'inputs':stored.inputs}
        encoded=codec.encode(payload)
        _Codec(codec.blobs.__getitem__,max_bytes=self.service.session.max_bytes).decode(encoded)
        raw=_json({'schema':'optimizer-checkpoint/1','payload':encoded})
        self.files[f'checkpoints/manifests/{oid}.json']=raw
        for digest,data in codec.blobs.items(): self.files[f'checkpoints/blobs/{digest}.npz']=data
        return {'hash':_hash(raw),'blobs':list(codec.blobs),'available':True,'resume_level':level,
                'type':type(value).__name__,'inputs':[ref.object_id for ref in stored.inputs],'context':stored.context,'configuration':stored.configuration}

    def _sync(self,session):
        index=self.index
        contexts={}
        for name,c in session._contexts.items():
            recipe=_owned(c.recipe); self.files.update(self._recipe_files(recipe))
            c.recipe=recipe
            codec=_Codec(max_bytes=session.max_bytes); encoded_recipe=codec.encode(recipe)
            self.files.update({f'checkpoints/blobs/{digest}.npz':data for digest,data in codec.blobs.items()})
            contexts[name]={'recipe':encoded_recipe,'configuration':c.configuration,
                            'head':c.head,'family':c.family}
            if self.track and (name not in index['contexts'] or c.head!=index['contexts'][name]['head'] or
                               c.configuration!=index['contexts'][name]['configuration']):
                state_id=uuid4().hex
                index['ancestry'][state_id]={'context':contexts[name],'parent':index['cursor'].get(name)}
                index['cursor'][name]=state_id; index['redo'][name]=[]
        index['contexts']=contexts; index['selected']=session._selected; index['bindings']=dict(session._bindings)
        for oid,stored in session._objects.items():
            if oid not in index['objects']: index['objects'][oid]=self._pack(oid,stored)
        # Recent and undo are preferences; active and pins are hard protection.
        protected=self.store.protected(index)
        for name,state_id in index['cursor'].items():
            for _ in range(index['policy']['undo_depth']):
                state=index['ancestry'].get(state_id)
                if not state: break
                if state['context']['head']: protected.add(state['context']['head'])
                state_id=state['parent']
        for path in index['redo'].values():
            for state_id in path[-index['policy']['undo_depth']:] if index['policy']['undo_depth'] else []:
                state=index['ancestry'].get(state_id)
                if state and state['context']['head']: protected.add(state['context']['head'])
        available=[oid for oid,e in index['objects'].items() if e['available']]
        recent=set(available[-index['policy']['keep_recent']:] if index['policy']['keep_recent'] else [])
        for oid in available:
            if oid not in protected|recent: index['objects'][oid]['available']=False
        index['generation']=self.store.index['generation']+1
        index['environment']=_environment()
        # Tombstones retain names without leaving pruned numerical payloads in RAM.
        session._pruned={oid for oid,entry in index['objects'].items() if not entry['available']}
        for oid in session._pruned: session._objects.pop(oid,None)
        for stored in session._objects.values():
            stored.bytes=_size(stored.value)+_size(stored.method_key)+_size(stored.recipe)+_size(stored.inputs)
        if session._retained_bytes()>session.max_bytes:
            raise _ServiceError('object_quota','Prepared durable data exceeds foreground retention.')
        self.store.commit(index,self.files,guard=self.guard)

    def execute(self,request):
        from optimizer.operations.service import OperationService, _summary
        session=self.service.session
        before=session.snapshot()
        self.guard=lambda: session._check_revision(before["revision"])
        try:
            if (self.store is not None and self.store._path('history.jsonl').exists()
                    and self.service._history is None and not self.store.read_only
                    and request.operation not in {'project.close','project.fork','project.list','project.status','storage.status','checkpoints.list'}):
                raise _ServiceError('history_owned','Unrecorded Python cannot mutate a history-owned experiment; open read-only or explicitly fork.')
            if request.operation in {'project.init','project.open','project.close','project.fork','project.list'}:
                if request.bind is not None or request.activate is not None or request.context is not None:
                    raise _ServiceError('invalid_arguments','Project lifecycle changes run without binding/activation/context.')
                method=self.methods[request.operation]; inspect.signature(method).bind(**request.arguments)
                result=method(**request.arguments)
                return OperationResponse(request.request_id,request.operation,'ok',result,state_before=before,
                    state_after=session.snapshot(),metadata={'summary':_summary(result)})
            store=self.require()
            self.staged=_clone(session); self.index=copy.deepcopy(store.index); self.files={}; self.changed=False; self.track=True
            if request.operation in self.methods:
                if request.activate is not None: raise _ServiceError('invalid_arguments','Lifecycle operations declare their activation semantics.')
                arguments=dict(request.arguments)
                if request.context is not None:
                    if 'context' in arguments: raise _ServiceError('invalid_arguments','Context supplied twice.')
                    arguments['context']=request.context
                method=self.methods[request.operation]; inspect.signature(method).bind(**arguments)
                result=method(**arguments); refs=()
                if request.bind:
                    if request.operation=='storage.plan': result['generation']=store.index['generation']+1
                    ref=self.staged._publish(result,revision=self.staged._revision,bind=request.bind)
                    refs=(ref,)
                response=OperationResponse(request.request_id,request.operation,'ok',result,refs,before,
                    self.staged.snapshot(),metadata={'summary':_summary(result)})
            else:
                response=OperationService(self.staged).execute(request)
                if not response.ok: return response
            with session._lock:
                session._check_revision(before['revision'])
                if self.staged._revision!=before['revision'] or self.changed:
                    self._sync(self.staged)
                    _adopt(session,self.staged)
                return OperationResponse(response.request_id,response.operation,response.status,response.result,response.object_refs,
                    before,session.snapshot(),metadata={**response.metadata,'durability':'persistent',
                    'store_generation':store.index['generation'],'durability_warning':store.warning,
                    'resume_level':store.index['objects'].get(response.object_refs[0].object_id,{}).get('resume_level') if response.object_refs else None})
        except BlockingIOError as exc:
            return self._error(request,before,'project_locked',str(exc))
        except Exception as exc:
            return self._error(request,before,exc.code if isinstance(exc,_ServiceError) else 'persistence_failed',str(exc))
        finally:
            self.staged=None; self.guard=None

    def _error(self,request,before,code,message):
        return OperationResponse(request.request_id,request.operation,'error',state_before=before,
                                 state_after=self.service.session.snapshot(),error={'code':code,'message':message})

    def init(self,name,root='.',max_bytes=536870912,keep_recent=20,undo_depth=10):
        if self.store: raise _ServiceError('project_open','Close the current experiment before opening another.')
        store=_Store.create(name,root,max_bytes=max_bytes,keep_recent=keep_recent,undo_depth=undo_depth)
        if self.service._history is not None:
            store=self.service._history.attach_store(store,created=True)
        self.store=store
        try:
            self.index=copy.deepcopy(store.index); self.files={}; self.track=True
            staged=_clone(self.service.session)
            self._sync(staged)
            _adopt(self.service.session,staged)
        except BaseException:
            self.store=None; store.close()
            import shutil
            if self.service._history is None: shutil.rmtree(store.path)
            raise
        return self.status()

    def _context(self,session,name,record,mode='exact'):
        recipe=self.store.codec(session.max_bytes).decode(record['recipe'])
        self._recipe_files(recipe)
        session.load(name,recipe['path'],parameters=recipe.get('arguments',{}),
                     factory_name=recipe['factory_name'],checkpoint=recipe.get('checkpoint',{}),select=False)
        for updates in recipe.get('secondary_updates',[]): session.update(context=name,updates=updates)
        c=session._contexts[name]
        c.configuration=record['configuration']; c.recipe=recipe
        if c.family!=record['family']: raise _ServiceError('incompatible_layout','Saved family changed.')
        c.head=record['head']
        if c.head:
            entry=self.store.index['objects'][c.head]
            payload=self.store.decode(c.head,session.max_bytes)
            if mode=='exact' and payload['environment']!=_environment():
                raise _ServiceError('environment_changed','Saved numerical state belongs to another source/environment.')
            if entry['resume_level']!='exact' and mode=='exact' and isinstance(payload['value'],(OptimizerResult,RunState,ShootingResult)):
                raise _ServiceError('unsupported_continuation','Saved state has physical resume only; open/restore with explicit physical mode.')
            value=payload['value']
            key=payload['method_key']
            if mode!='exact':
                value=_physical(value); key=None
                expected='controls_only' if hasattr(value,'as_matrix') else 'variables_only'
                if mode!=expected: raise _ServiceError('invalid_arguments','Physical mode does not match saved family.')
                c.head=uuid4().hex
            _compatible(c.layout,value)
            session._objects[c.head]=_Stored(value,_size(value)+_size(key)+_size(recipe),name,c.configuration,key,recipe,payload.get("inputs",()))
            c.method_key=key

    def open(self,path,read_only=False,mode='exact',migrate=False):
        if self.store: raise _ServiceError('project_open','Close current experiment first.')
        if self.service.session.snapshot()['contexts'] or self.service.session.snapshot()['bindings']:
            raise _ServiceError('unsaved_session','Opening would abandon foreground objects; use a fresh service or close an existing project.')
        if type(read_only) is not bool or mode not in ('exact','controls_only','variables_only'):
            raise _ServiceError('invalid_arguments','read_only must be bool; choose an explicit restore mode.')
        if type(migrate) is not bool: raise _ServiceError('invalid_arguments','migrate must be bool.')
        store=_Store(path,read_only=read_only)
        if self.service._history is not None:
            store=self.service._history.attach_store(store,migrate=migrate)
        elif store._path('history.jsonl').exists() and not read_only:
            store.close()
            raise _ServiceError('history_owned','Open a history-owned experiment read_only=True, or use the terminal frontend.')
        self.store=store
        try:
            if mode=='exact' and store.index['environment']!=_environment():
                raise _ServiceError('environment_changed','Exact restoration requires the recorded numerical environment/source.')
            session=Session(max_bytes=self.service.session.max_bytes,max_objects=self.service.session.max_objects)
            for name,record in store.index['contexts'].items(): self._context(session,name,record,mode)
            for name,oid in store.index['bindings'].items():
                entry=store.index['objects'].get(oid)
                if entry and entry['available'] and oid not in session._objects:
                    data=store.decode(oid,session.max_bytes)
                    session._objects[oid]=_Stored(data['value'],_size(data['value'])+_size(data['method_key'])+_size(data['recipe']),
                        data['context'],data['configuration'],data['method_key'] if mode=='exact' else None,data['recipe'],data.get('inputs',()))
                if entry: session._bindings[name]=oid
            session._pruned={oid for oid,entry in store.index['objects'].items() if not entry['available']}
            session._selected=store.index['selected']
            if session._retained_bytes()>session.max_bytes or len(session._objects)>session.max_objects:
                raise _ServiceError('object_quota','Restored named data exceeds foreground retention; open with a larger Session budget.')
            session._revision=self.service.session._revision+1
            if mode!='exact' and not read_only:
                self.index=copy.deepcopy(store.index); self.files={}; self.track=True
                self._sync(session)
            _adopt(self.service.session,session)
            return self.status()
        except BaseException:
            store.close(); self.store=None; raise

    def close(self):
        if self.store:
            self.store.close(); self.store=None
            session=self.service.session
            empty=Session(max_bytes=session.max_bytes,max_objects=session.max_objects)
            empty._revision=session._revision+1; _adopt(session,empty)
        return {'project':'closed','data':'preserved'}

    def fork(self,name,root=None):
        store=self.require(); fork=store.clone(name,root)
        if self.service._history is not None:
            fork=self.service._history.attach_store(fork,created=True)
        store.close(); self.store=fork
        return self.status()

    def list_projects(self,root='.'):
        parent=Path(root).expanduser().resolve()/'.optimizer'
        candidates=[]
        if parent.is_dir():
            for path in sorted(parent.iterdir())[:256]:
                if path.is_dir() and not path.is_symlink() and (path/'project.json').is_file():
                    candidates.append({'name':path.name,'path':str(path)})
        return {'projects':candidates,'limit':256}

    def status(self):
        store=self.require()
        return {'name':store.project['name'],'path':str(store.path),'project_id':store.project['id'],
                'generation':store.index['generation'],'read_only':store.read_only,'stored_bytes':store.usage(),
                'policy':store.index['policy'],'protected':sorted(store.protected()),'reserved_bytes':0,
                **store.reclaim([oid for oid,e in store.index['objects'].items() if e['available'] and oid not in store.protected()]),
                'undo_available':self._undo_available(store.index),
                'redo_available':{name:sum(1 for sid in reversed(path) if (not store.index['ancestry'][sid]['context']['head'] or store.index['objects'][store.index['ancestry'][sid]['context']['head']]['available'])) for name,path in store.index['redo'].items()},
                'durability_warning':store.warning,
                'resume_levels':{name:store.index['objects'].get(c['head'],{}).get('resume_level') for name,c in store.index['contexts'].items()}}

    @staticmethod
    def _undo_available(index):
        result={}
        for name,cursor in index['cursor'].items():
            count=0
            seen=set()
            while cursor and cursor not in seen:
                seen.add(cursor)
                state=index['ancestry'].get(cursor)
                cursor=state.get('parent') if state else None
                previous=index['ancestry'].get(cursor)
                if not previous: break
                head=previous['context']['head']
                if head and not index['objects'].get(head,{}).get('available'): break
                count+=1
            result[name]=count
        return result

    def _identity(self,value):
        if isinstance(value,ObjectRef): value=self.staged.get(value)
        if isinstance(value,dict): value=value.get('checkpoint_id')
        if not isinstance(value,str): raise _ServiceError('invalid_arguments','Checkpoint requires a name, ID or save-result reference.')
        identity=self.index['bookmarks'].get(value,value)
        if identity not in self.index['objects']: raise _ServiceError('checkpoint_missing','Unknown checkpoint name or ID.')
        return identity

    def save(self,name,mode='exact',context=None):
        c=self.staged._context(context)
        if not c.head: raise _ServiceError('missing_state','Initialize numerical state first.')
        if mode not in ('exact','controls_only','variables_only'): raise _ServiceError('invalid_arguments','Unknown save mode.')
        value=self.staged._objects[c.head].value
        physical=_physical(value)
        path=() if mode=='exact' or physical is value else ('controls',) if hasattr(physical,'as_matrix') else ('variables',)
        return self.save_object(ObjectRef(c.head,path),name=name,kind='full_state' if mode=='exact' else 'controls' if mode=='controls_only' else 'variables')

    def save_object(self,value,name=None,kind='report',pin=True):
        if type(pin) is not bool or kind not in ('full_state','controls','variables','report','input'):
            raise _ServiceError('invalid_arguments','Invalid checkpoint kind or pin flag.')
        source=None
        if isinstance(value,ObjectRef):
            source=self.staged._objects[self.staged._resolve_id(value)]
            actual=self.staged.get(value)
        else: actual=value
        if kind=='full_state' and (source is None or self._level(source)!='exact'):
            raise _ServiceError('unsupported_continuation','Exact save requires a certified method and checkpoint={stateless:True, dependencies:[...]}.')
        if kind=='full_state' and not isinstance(actual,(RunState,OptimizerResult,ShootingResult)):
            raise _ServiceError('invalid_state','Full checkpoint requires complete native method state.')
        if kind in ('controls','variables'):
            from optimizer.controls import Controls
            from optimizer.system_olss import ShootingVariables
            expected=Controls if kind=='controls' else ShootingVariables
            if not isinstance(actual,expected): raise _ServiceError('invalid_arguments','Checkpoint kind does not match physical value.')
        if kind=='report' and _physical(actual) is not None:
            raise _ServiceError('invalid_arguments','Use the explicit physical/full-state kind for numerical state.')
        ref=self.staged._publish(actual,revision=self.staged._revision,
            method_key=source.method_key if source and kind=='full_state' else None,
            inputs=source.inputs if source else (),
            extra_bytes=_size(source.recipe)-_size(None) if source else 0)
        stored=self.staged._objects[ref.object_id]
        if source:
            stored.context,stored.configuration,stored.recipe=source.context,source.configuration,_owned(source.recipe)
            stored.bytes+=_size(source.recipe)-_size(None)
        if name is not None:
            if not isinstance(name,str) or not name.strip(): raise _ServiceError('invalid_arguments','Checkpoint name must be nonempty.')
            self.index['bookmarks'][name]=ref.object_id
        if pin: self.index['pins'].append(ref.object_id)
        self.changed=True
        return {'checkpoint_id':ref.object_id,'kind':kind,'resume_level':self._level(stored),'name':name,'pinned':pin}

    def list_checkpoints(self,limit=100):
        if type(limit) is not int or not 1<=limit<=256: raise _ServiceError('invalid_arguments','limit must be 1–256.')
        return {'checkpoints':[{'checkpoint_id':oid,**entry,'pinned':oid in self.index['pins']}
                               for oid,entry in list(self.index['objects'].items())[-limit:]],'bookmarks':self.index['bookmarks']}

    def load(self,checkpoint):
        identity=self._identity(checkpoint)
        return self.store.decode(identity,self.staged.max_bytes)['value']

    def restore(self,checkpoint,mode='exact',context=None):
        if mode not in ('exact','controls_only','variables_only'): raise _ServiceError('invalid_arguments','Unknown restore mode.')
        identity=self._identity(checkpoint)
        data=self.store.decode(identity,self.staged.max_bytes)
        if data['context'] is None: raise _ServiceError('invalid_state','Checkpoint has no system context.')
        name=context or data['context']
        if mode=='exact' and data['resume_level']!='exact': raise _ServiceError('unsupported_continuation','Checkpoint cannot restore exact method state.')
        if mode=='exact' and data['environment']!=_environment(): raise _ServiceError('environment_changed','Numerical environment changed.')
        if mode=='exact':
            old=self.staged._contexts.pop(name,None)
            self._context(self.staged,name,{'recipe':_Codec().encode(data['recipe']),'configuration':data['configuration'],
                'head':identity,'family':old.family if old else ('olgs' if hasattr(_physical(data['value']),'as_matrix') else 'olss')},mode)
            c=self.staged._contexts[name]
            ref=self.staged._publish(data['value'],revision=self.staged._revision,context=c,activate=True,method_key=data['method_key'])
        else:
            c=self.staged._context(name); value=_physical(data['value'])
            expected='controls_only' if hasattr(value,'as_matrix') else 'variables_only'
            if mode!=expected: raise _ServiceError('invalid_arguments','Physical mode does not match saved family.')
            ref=self.staged.use(value,context=name)
        self.staged._selected=name; self.changed=True
        return {'restored':identity,'state':ref.object_id,'mode':mode,'context':name}

    def _move(self,steps,context,redo):
        if type(steps) is not int or steps<1: raise _ServiceError('invalid_arguments','steps must be positive integer.')
        c=self.staged._context(context); name=c.name; cursor=self.index['cursor'].get(name)
        path=self.index['redo'].setdefault(name,[])
        for _ in range(steps):
            if redo:
                if not path: raise _ServiceError('redo_unavailable','No unambiguous redo path; restore an explicit checkpoint.')
                cursor=path.pop()
            else:
                state=self.index['ancestry'].get(cursor)
                parent=state['parent'] if state else None
                if not parent: raise _ServiceError('undo_unavailable','Requested undo range is not retained.')
                path.append(cursor); cursor=parent
        record=self.index['ancestry'][cursor]['context']
        self.staged._contexts.pop(name)
        self._context(self.staged,name,record)
        self.index['cursor'][name]=cursor; self.track=False; self.changed=True
        self.staged._revision+=1
        return {'context':name,'state':cursor,'head':record['head']}

    def undo(self,steps=1,context=None): return self._move(steps,context,False)
    def redo(self,steps=1,context=None): return self._move(steps,context,True)
    def pin(self,checkpoint):
        identity=self._identity(checkpoint); self.store.manifest(identity)
        if identity not in self.index['pins']: self.index['pins'].append(identity)
        self.changed=True; return {'checkpoint_id':identity,'pinned':True}
    def unpin(self,checkpoint):
        identity=self._identity(checkpoint)
        self.index['pins']=[oid for oid in self.index['pins'] if oid!=identity]
        self.changed=True; return {'checkpoint_id':identity,'pinned':False}

    def configure(self,max_bytes=None,keep_recent=None,undo_depth=None):
        policy=dict(self.index['policy'])
        for name,value in (('max_bytes',max_bytes),('keep_recent',keep_recent),('undo_depth',undo_depth)):
            if value is not None: policy[name]=value
        self.index['policy']=_Store.policy(**policy); self.changed=True
        return policy

    def plan(self,action='prune',ids=None):
        if action not in ('prune','delete','purge'): raise _ServiceError('invalid_arguments','Unknown cleanup action.')
        protected=self.store.protected(self.index)
        eligible=[oid for oid,e in self.index['objects'].items() if e['available'] and oid not in protected]
        targets=eligible if action in ('prune','purge') else [self._identity(oid) for oid in (ids or [])]
        if any(oid not in eligible for oid in targets): raise _ServiceError('protected_payload','Unknown/unavailable/protected cleanup target.')
        return {'project_id':self.store.project['id'],'generation':self.store.index['generation'],
                'action':action,'targets':[{ 'id':oid,'hash':self.index['objects'][oid]['hash']} for oid in targets],
                'loss':{'bookmarks':[name for name,oid in self.index['bookmarks'].items() if oid in targets],
                        'ancestry':[sid for sid,state in self.index['ancestry'].items() if state['context']['head'] in targets],
                        'dependent_reports':[oid for oid,entry in self.index['objects'].items() if set(entry.get('inputs',[])) & set(targets)],
                        'message':'Selected payloads and dependent undo/restore paths become unavailable; metadata remains.'},
                **self.store.reclaim(targets,self.index)}

    def apply(self,plan):
        if isinstance(plan,ObjectRef): plan=self.staged.get(plan)
        if not isinstance(plan,dict) or set(plan)!={'project_id','generation','action','targets','loss','reclaimable_bytes','removed_blobs','shared_blobs_kept'}:
            raise _ServiceError('invalid_arguments','Expected concrete storage plan.')
        if plan['project_id']!=self.store.project['id'] or plan['generation']!=self.store.index['generation']:
            raise _ServiceError('stale_plan','Cleanup plan no longer matches store generation.')
        protected=self.store.protected(self.index)
        for target in plan['targets']:
            oid=target['id']; entry=self.index['objects'].get(oid)
            if oid in protected or not entry or not entry['available'] or entry['hash']!=target['hash']:
                raise _ServiceError('protected_payload','Cleanup dependencies/identity changed.')
        for target in plan['targets']: self.index['objects'][target['id']]['available']=False
        self.changed=True
        return {'removed':[target['id'] for target in plan['targets']]}
