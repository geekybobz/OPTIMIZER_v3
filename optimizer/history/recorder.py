"""One frontend statement stages all child operations before publication."""
from __future__ import annotations
import copy
import inspect
from collections import deque
from dataclasses import asdict
from uuid import uuid4
from optimizer.checkpoints.codec import _json, _hash
from optimizer.operations.contracts import OperationResponse, _ServiceError
from optimizer.persistence.controller import _clone, _adopt
from optimizer.persistence.store import _Store, _environment
from optimizer.history.journal import _Journal
from optimizer.cli.display import _describe
from optimizer.serialization import to_plain

class _StagedStore(_Store):
    def __init__(self, real):
        self.__dict__=dict(real.__dict__)
        self.real=real; self.index=copy.deepcopy(real.index);self.files={}
    def read(self,relative,limit=64*1024*1024):
        self._path(relative)
        if relative in self.files:
            data=self.files[relative]
            if len(data)>limit:raise _ServiceError('codec_limit','Prepared file exceeds read budget.')
            return data
        return self.real.read(relative,limit)
    def commit(self,index,files,guard=None,**kwargs):
        if self.read_only:raise _ServiceError('read_only','Experiment is open read-only.')
        if index['generation']!=self.index['generation']+1:raise _ServiceError('revision_conflict','Staged generation changed.')
        if guard:guard()
        self.files.update(files)
        # Children share one outer generation. This also keeps a bound cleanup
        # plan valid after a nested constructor inside the same statement.
        index['generation']=self.real.index['generation']
        self.index=index
    def close(self):pass
    def clone(self,name,root=None):return self.real.clone(name,root)

class _Recorder:
    def __init__(self,service):
        self.service=service;self.session_id=uuid4().hex
        self.memory=deque(maxlen=256);self.memory_bytes=0
        self.active=None;self.command=None;self.targets=[];self.events=[]
        self.methods={'history.list':self.list,'history.show':self.show,
                      'history.replay_plan':self.replay_plan,'history.replay':self.replay}
        service._history=self

    def descriptors(self):
        return [dict(id=n,name=n.split('.')[-1],classification='lifecycle',effect='history',
                     signature=str(inspect.signature(m)),reason='Shared CLI/MCP frontend recorder.') for n,m in self.methods.items()]

    def execute(self,request,service):
        before=service.session.snapshot()
        try:
            if request.activate is not None or request.context is not None:
                raise _ServiceError('invalid_arguments','History operations do not activate a numerical context.')
            args=service._resolve(request.arguments)
            method=self.methods[request.operation];inspect.signature(method).bind(**args)
            result=method(**args)
            refs=(service.register_reference(request.bind,result),) if request.bind else ()
            return OperationResponse(request.request_id,request.operation,'ok',result,refs,before,service.session.snapshot(),metadata={'summary':_describe(result)})
        except Exception as exc:
            return OperationResponse(request.request_id,request.operation,'error',state_before=before,state_after=service.session.snapshot(),error={'code':getattr(exc,'code','invalid_arguments'),'message':str(exc)})

    def capture(self,response,request=None):
        def refs(value):
            from optimizer.operations import ObjectRef
            if isinstance(value,ObjectRef):return asdict(value)
            if isinstance(value,dict):return {str(k):refs(v) for k,v in value.items()}
            if isinstance(value,(tuple,list)):return [refs(v) for v in value]
            if value is None or type(value) in (str,bool,int,float):return value
            return to_plain(_describe(value))
        summary=refs(response.metadata.get('summary',{}))
        if len(_json(summary))>8192:summary={'omitted':'Inspect retained object references for the complete result.'}
        self.events.append(dict(operation=response.operation,arguments=refs(request.arguments) if request else {},
                                effective_arguments=refs(response.metadata.get('effective_arguments',{})),
                                object_refs=[asdict(ref) for ref in response.object_refs],
                                context=refs(request.context) if request else None,bind=refs(request.bind) if request else None,
                                activate=refs(request.activate) if request else None,status=response.status,error=response.error,
                                summary=summary))

    def attach_store(self,store,*,created=False,migrate=False):
        if store.read_only:return _StagedStore(store)
        path=store._path('history.jsonl')
        journal=_Journal(store)
        if journal.pending:
            journal.recover()
        if self.capacity_override is not None and store is self.service._persistence.store:
            journal.capacity_override=self.capacity_override
        if not path.exists():
            if not created and not migrate:
                store.close()
                raise _ServiceError('history_migration_required','This is an index-owned experiment. Use project.open(..., migrate=True) to record an explicit baseline.')
            journal.append('baseline',dict(index=copy.deepcopy(store.index),origin=store.project['origin'] if created else {'migration':'index/1','generation':store.index['generation']}))
        try:
            journal.begin(self.command,self.command_id,self.session_id)
        except BaseException:
            if created:store.close()
            raise
        self.targets.append((store,journal))
        return _StagedStore(store)

    def run(self,raw,expression,callback,*,before=None,origin='command'):
        root=self.service
        if self.active is not None:raise _ServiceError('reentrant_command','A frontend statement is already running.')
        with root._writer:
            with root.session._lock:
                before=root.session.snapshot() if before is None else before
                staged=copy.copy(root);staged.session=_clone(root.session)
                staged._job_actions=[];staged._defer_jobs=True
            from optimizer.persistence.controller import _Persistence
            staged._persistence=_Persistence(staged);staged._history=self
            self.active=staged;self.command_id=uuid4().hex
            self.capacity_override=self._capacity_override(raw,root._persistence.store)
            self.command=dict(raw=raw,expression=expression,state_before=self._state(before),environment=_environment(),origin=origin)
            self.targets=[];self.events=[]
            original=root._persistence.store
            try:
                if original is not None:
                    staged._persistence.store=self.attach_store(original)
                result=callback(staged)
                good=all(event['status']=='ok' for event in self.events)
                status='ok' if good else 'error'
                if result==130:status='interrupted';good=False
                error=next((e['error'] for e in reversed(self.events) if e['error']),None)
                if not good:
                    outcome=self._outcome(status,before,before,error)
                    for real,journal in self.targets:journal.outcome(outcome,self.command_id,self.session_id)
                else:
                    with root.session._lock:
                        root.session._check_revision(before['revision'])
                        outcome=self._outcome('ok',before,staged.session.snapshot(),None)
                        destination=staged._persistence.store
                        if destination is not None and not destination.read_only:
                            real=destination.real
                            if destination.index!=real.index:
                                index=copy.deepcopy(destination.index)
                                index['generation']=real.index['generation']+1
                                # A retained cleanup plan belongs to this single outer commit.
                                real.commit(index,destination.files,
                                            guard=lambda:root.session._check_revision(before['revision']),
                                            history=(next(j for r,j in self.targets if r is real),outcome,self.command_id,self.session_id))
                            else:
                                next(j for r,j in self.targets if r is real).outcome(outcome,self.command_id,self.session_id)
                        for real,journal in self.targets:
                            if destination is None or real is not getattr(destination,'real',None):
                                journal.outcome(outcome,self.command_id,self.session_id)
                        _adopt(root.session,staged.session)
                        root._persistence.store=destination.real if destination is not None else None
                        if original is not None and root._persistence.store is not original:original.close()
                        if not root._jobs.durable and any(e['operation']=='session.exit' for e in self.events):
                            staged._job_actions.append(('shutdown',None))
                        root._jobs.commit(staged._job_actions)
                self._remember(outcome)
                return result
            except BaseException as exc:
                outcome=self._outcome('interrupted' if isinstance(exc,KeyboardInterrupt) else 'error',before,before,
                                      {'code':getattr(exc,'code','history_failed'),'type':type(exc).__name__,'message':str(exc)})
                # A full/failed intent never runs the command. An acknowledged
                # intent has its failure reserve; no numerical state is adopted.
                for real,journal in self.targets:
                    if self.command_id in journal.pending:journal.outcome(outcome,self.command_id,self.session_id)
                self._remember(outcome)
                raise
            finally:
                for real,journal in self.targets:
                    if real is not root._persistence.store:real.close()
                self.active=None
                self.command=None

    @staticmethod
    def _capacity_override(raw,store):
        """Only an explicit literal quota increase can fund its own recording."""
        if store is None or store.read_only:return None
        import ast
        from optimizer.cli.language import _parse,_operation
        try:
            statements=_parse(raw.strip())
            if len(statements)!=1:return None
            expression,binding,statement=statements[0]
            if binding or not isinstance(expression,ast.Call) or _operation(expression.func)!='storage.configure' or expression.args:return None
            if len(expression.keywords)!=1 or expression.keywords[0].arg!='max_bytes':return None
            value=expression.keywords[0].value
            if not isinstance(value,ast.Constant) or type(value.value) is not int:return None
            return value.value if value.value>store.index['policy']['max_bytes'] else None
        except Exception:return None

    def _outcome(self,status,before,after,error):
        operations=self.events[:128]
        if status!='ok':
            operations=[{k:event[k] for k in ('operation','object_refs','context','bind','activate','status')} for event in operations]
            if len(_json(operations))>32768:
                operations=[{'omitted':len(operations),'sha256':_hash(_json(operations))}]
        if error is not None:
            error={**error,'message':str(error.get('message',''))[:2048]}
        summary=self.events[-1]['summary'] if self.events else None
        if summary is not None and len(_json(summary))>8192:summary={'omitted':'Result summary exceeds 8 KiB; inspect retained object references.'}
        return dict(status=status,state_before=self._state(before),state_after=self._state(after),operations=operations,error=error,
                    summary=summary,index=None)

    @staticmethod
    def _state(snapshot):
        value={k:v for k,v in snapshot.items() if k!='contexts'} | {
            'contexts':{n:{k:c[k] for k in ('head','configuration','family')} for n,c in snapshot.get('contexts',{}).items()}}
        if len(_json(value))>32768:
            return dict(revision=snapshot['revision'],selected=str(snapshot.get('selected'))[:128],
                        context_count=len(snapshot.get('contexts',{})),binding_count=len(snapshot.get('bindings',{})),
                        sha256=_hash(_json(value)),omitted='Inspect the committed index for full state metadata.')
        return value

    def _remember(self,outcome):
        value=dict(command_id=self.command_id,session_id=self.session_id,raw=self.command['raw'],
                   expression=self.command['expression'],status=outcome['status'],summary=outcome['summary'],error=outcome['error'])
        size=len(_json(value))
        while self.memory and (self.memory_bytes+size>1024*1024 or len(self.memory)>=256):
            self.memory_bytes-=len(_json(self.memory.popleft()))
        if size<=1024*1024:self.memory.append(value);self.memory_bytes+=size

    def _store(self):return (self.active or self.service)._persistence.store

    def list(self,limit=50,before=None):
        store=self._store()
        if store is None:
            if type(limit) is not int or not 1<=limit<=200:raise _ServiceError('invalid_arguments','history limit must be 1..200.')
            rows=list(self.memory)
            if before is not None:
                found=next((i for i,r in enumerate(rows) if r['command_id']==before),None)
                if found is None:raise _ServiceError('history_missing','Unknown history command.')
                rows=rows[:found]
            return {'durability':'ephemeral','commands':rows[-limit:],'retention':'256 commands / 1 MiB'}
        real=getattr(store,'real',store)
        journal=_Journal(real)
        return {'durability':'persistent' if not real.read_only else 'read_only',
                'commands':journal.commands(limit,before)}

    def show(self,command_id):
        store=self._store()
        if store is None:
            row=next((r for r in self.memory if r['command_id']==command_id),None)
            if row is None:raise _ServiceError('history_missing','Unknown history command.')
            return row
        pair=[]
        for row in _Journal(getattr(store,'real',store)).rows():
            if row['command_id']==command_id:
                row=copy.deepcopy(row)
                if row['kind']=='outcome' and row['data'].get('index') is not None:
                    index=row['data'].pop('index');row['data']['commit']=dict(generation=index['generation'],heads={n:c['head'] for n,c in index['contexts'].items()},cursor=index['cursor'])
                pair.append(row)
        if not pair:raise _ServiceError('history_missing','Unknown history command.')
        return {'records':pair}

    def replay_plan(self,target='latest',mode='checkpoint'):
        if mode not in {'checkpoint','commands'}:raise _ServiceError('invalid_arguments','Replay mode must be checkpoint or commands.')
        store=self._store()
        if store is None:raise _ServiceError('missing_project','Replay requires a named experiment.')
        real=getattr(store,'real',store);journal=_Journal(real)
        index=None;steps=[];selected=None;baseline=None;issues=[]
        intentions={};selected_command=None;recipes={};history_bindings=set()
        bookmark_id=real.index['bookmarks'].get(target)
        for row in journal.rows():
            if row['kind']=='baseline':baseline=index=row['data']['index']
            elif row['kind']=='submitted':intentions[row['command_id']]=row
            elif row['kind']=='outcome':
                intent=intentions.pop(row['command_id']);data=row['data']
                if data['status']=='ok':
                    index=data.get('index') or index
                    raw=intent['data']['raw'];operations=data.get('operations',[])
                    for c in index['contexts'].values():
                        recipe=c['recipe'];recipes[_hash(_json(recipe))]=recipe
                    history_bindings.update(e['bind'] for e in operations if e['operation'].startswith('history.') and e.get('bind'))
                    if operations and not any(e['operation'].startswith(('project.','history.')) or e['operation']=='session.exit' for e in operations):steps.append(raw)
                selected_command=row['command_id']
                if target==row['command_id'] or (bookmark_id is not None and index['bookmarks'].get(target)==bookmark_id):
                    selected=index;break
        if target=='latest':
            selected=index
            # A retained preview is itself a command. Freeze the preceding
            # outcome so its own assignment does not move the replay target.
            target=selected_command
        if selected is None:raise _ServiceError('history_missing','Replay target must be latest, a command ID, or a saved bookmark.')
        candidate=copy.deepcopy(selected)
        if target in candidate['bookmarks']:
            oid=candidate['bookmarks'][target];entry=candidate['objects'][oid];context=entry['context']
            if context:candidate['contexts'][context]['head']=oid
        for oid,entry in candidate['objects'].items():
            current=real.index['objects'].get(oid,{})
            if entry['available'] and not current.get('available'):
                entry['available']=False
                issues.append({'code':'payload_pruned','object_id':oid,'required':mode=='checkpoint' and oid in real.protected(candidate)})
        for name,c in candidate['contexts'].items():
            if c['head'] and not candidate['objects'][c['head']]['available']:continue
            if mode=='checkpoint' and c['head'] and candidate['objects'][c['head']]['resume_level']!='exact':
                issues.append({'code':'physical_only','context':name,'required':True})
            recipe=real.codec(self.service.session.max_bytes).decode(c['recipe'])
            try:
                from optimizer.persistence.controller import _Persistence
                _Persistence(self.service)._recipe_files(recipe)
            except Exception as exc:issues.append({'code':getattr(exc,'code','source_changed'),'required':True,'message':str(exc)})
        if candidate['environment']!=_environment():issues.append({'code':'environment_changed','required':True})
        if len(steps)>512:issues.append({'code':'replay_limit','required':True})
        if mode=='commands' and (baseline['contexts'] or baseline['bindings']):issues.append({'code':'nonempty_baseline','required':True})
        if mode=='commands':
            import ast
            from optimizer.cli.language import _parse, _operation
            import pathlib
            for encoded in recipes.values():
                try:
                    recipe=real.codec(self.service.session.max_bytes).decode(encoded)
                    from optimizer.persistence.controller import _Persistence
                    _Persistence(self.service)._recipe_files(recipe)
                except Exception as exc:
                    issues.append({'code':getattr(exc,'code','source_changed'),'required':True,'message':str(exc)})
            # Every source load, immutable literal reference and stochastic call
            # is checked, including commands whose context was later released.
            for raw in steps:
                for expression,binding,statement in _parse(raw.strip()):
                    for node in ast.walk(expression):
                        if isinstance(node,ast.Name) and node.id in history_bindings:
                            issues.append({'code':'history_binding_dependency','required':True,'name':node.id})
                        if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value in candidate['objects']:
                            issues.append({'code':'literal_object_identity','required':True})
                        if not isinstance(node,ast.Call):continue
                        operation=_operation(node.func)
                        if operation.startswith('jobs.'):
                            issues.append({'code':'asynchronous_replay_requires_checkpoint','operation':operation,'required':True})
                        keywords={k.arg:k.value for k in node.keywords}
                        if 'cma_es' in operation or operation.startswith('guesses.random') or operation.startswith('random'):
                            seed=keywords.get('seed')
                            if not isinstance(seed,ast.Constant) or type(seed.value) is not int:
                                issues.append({'code':'stochastic_seed_required','operation':operation,'required':True})
                        if operation=='session.load':
                            path=keywords.get('path') or (node.args[1] if len(node.args)>1 else None)
                            if not isinstance(path,ast.Constant) or not isinstance(path.value,str):
                                issues.append({'code':'source_not_literal','required':True})
                            elif not pathlib.Path(path.value).expanduser().is_file():
                                issues.append({'code':'source_missing','required':True,'path':path.value})
                        # Trusted registered Python callbacks cannot be reconstructed
                        # from a name alone; source loading is handled separately.
                        if operation=='session.attach':issues.append({'code':'live_system_baseline','required':True})
            # Check declared sources for every historical load, including contexts
            # released later; an index-only baseline cannot invent those commands.
            for row in journal.rows():
                if row['kind']=='submitted' and row['data']['environment']!=_environment():
                    issues.append({'code':'environment_changed','required':True});break
                if row['kind']=='outcome' and row['command_id']==selected_command:break
        core=dict(schema='optimizer-replay-plan/1',project_id=real.project['id'],source=str(real.path),
                  target=target,mode=mode,index=candidate,steps=steps,issues=issues,
                  executable=not any(x['required'] for x in issues),
                  work={'statements':len(steps),'numerical_work':'recomputed' if mode=='commands' else 'none'},
                  history_hash=journal.digest)
        return {**core,'hash':_hash(_json(core))}

    def replay(self,plan,project,root=None):
        if not isinstance(plan,dict):raise _ServiceError('invalid_arguments','Use a retained replay plan.')
        core={k:v for k,v in plan.items() if k!='hash'}
        if plan.get('hash')!=_hash(_json(core)) or core.get('schema')!='optimizer-replay-plan/1':raise _ServiceError('invalid_replay_plan','Replay plan was changed or has an unknown version.')
        store=self._store();real=getattr(store,'real',store)
        if real is None or core['project_id']!=real.project['id']:raise _ServiceError('invalid_replay_plan','Plan belongs to another experiment.')
        current_plan=self.replay_plan(core['target'],core['mode'])
        if current_plan['index']!=core['index'] or current_plan['steps']!=core['steps']:
            raise _ServiceError('stale_plan','Replay target changed or the plan contains a fabricated checkpoint index.')
        if not current_plan['executable']:
            raise _ServiceError('replay_unavailable','Replay dependencies changed after planning.')
        if not core['executable']:raise _ServiceError('replay_unavailable','Review replay plan issues before execution.')
        if core['index']['environment']!=_environment():raise _ServiceError('environment_changed','Replay environment changed.')
        for oid,entry in core['index']['objects'].items():
            if entry['available'] and not real.index['objects'].get(oid,{}).get('available'):
                raise _ServiceError('payload_pruned','Replay payload was pruned after planning.')
        destination_root=str(real.path.parent.parent) if root is None else root
        if core['mode']=='commands':
            from optimizer.cli.language import _Terminal
            terminal=_Terminal()
            import json
            code=terminal.submit('opt.project.init('+json.dumps(project)+', root='+json.dumps(destination_root)+');')
            if code:raise _ServiceError('replay_failed','Could not initialize the new replay experiment.')
            try:
                for position,raw in enumerate(core['steps']):
                    if terminal.submit(raw.strip()):raise _ServiceError('replay_failed',f'Replay stopped at step {position+1}; destination history preserves the attempt.')
                return {'project':str(terminal.service._persistence.store.path),'mode':'commands','statements':len(core['steps'])}
            finally:terminal.service._persistence.close()
        destination=_Store.create(project,destination_root,**real.index['policy'],origin={'project_id':real.project['id'],'history_hash':core['history_hash'],'target':core['target']})
        try:
            index=copy.deepcopy(core['index']);index['project_id']=destination.project['id'];index['generation']=1
            needed={f'checkpoints/manifests/{oid}.json' for oid,e in index['objects'].items() if e['available']}
            blobs={b for e in index['objects'].values() if e['available'] for b in e['blobs']}|real._recipe_blobs(index)
            needed|={f'checkpoints/blobs/{b}.npz' for b in blobs}
            # Source capsules are immutable; copy them, then collect unreferenced
            # capsules after adopting the selected checkpoint index.
            needed|={p.relative_to(real.path).as_posix() for p in real._files() if p.relative_to(real.path).parts[0]=='inputs'}
            destination.commit(index,{p:real.read(p) for p in needed})
            _Journal(destination).append('baseline',dict(index=copy.deepcopy(index),origin=destination.project['origin']))
            return {'project':str(destination.path),'mode':'checkpoint','recalculated':False}
        finally:destination.close()
