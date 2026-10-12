"""Single-writer, content-addressed experiment storage and atomic indexes."""
from __future__ import annotations
import copy
import importlib.metadata
import os
import platform
import re
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4
import numpy as np
from optimizer.checkpoints.codec import _Codec, _hash, _json
from optimizer.operations.contracts import _ServiceError

_ID = re.compile(r'^[a-f0-9]{32,64}$')
_NAME = re.compile(r'^[A-Za-z][A-Za-z0-9_-]{0,63}$')

def _environment():
    source=Path(__file__).resolve().parents[1]
    digest=_hash(b''.join(p.relative_to(source).as_posix().encode()+p.read_bytes()
                         for p in sorted(source.rglob('*.py'))))
    return {'python': platform.python_version(), 'numpy': np.__version__, 'machine': platform.machine(),
            'platform': platform.system(), 'optimizer_source':digest,
            'scipy': importlib.metadata.version('scipy') if importlib.util.find_spec('scipy') else None}

class _CommittedWriteError(OSError):
    """Replacement succeeded, but directory fsync failed; never undo the index."""

def _atomic(path, payload):
    fd, temporary=tempfile.mkstemp(prefix='.prepare-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary,path)
        try:
            directory=os.open(path.parent,os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
        except OSError as exc:
            raise _CommittedWriteError(str(exc)) from exc
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

class _Store:
    def __init__(self, path, *, read_only=False):
        import fcntl
        self.path=Path(path).expanduser().resolve(strict=True)
        self.read_only=read_only
        self.warning=None
        for folder in ('runtime','inputs','checkpoints','checkpoints/manifests','checkpoints/blobs'):
            if (self.path/folder).is_symlink() or not (self.path/folder).is_dir():
                raise _ServiceError('checkpoint_corrupt','Missing or symlinked store directory.')
        if (self.path/'runtime/writer.lock').is_symlink():
            raise _ServiceError('checkpoint_corrupt','Symlinked writer lease.')
        self.lock=(self.path/'runtime/writer.lock').open('a+b')
        try:
            fcntl.flock(self.lock.fileno(), (fcntl.LOCK_SH if read_only else fcntl.LOCK_EX)|fcntl.LOCK_NB)
            self.project=self.read_json('project.json')
            try:
                self.index=self.read_json('session.json')
            except _ServiceError:
                if not self._path('history.jsonl').exists(): raise
                self.index={'generation':-1}
            if self._path('history.jsonl').exists():
                from optimizer.history.journal import _Journal
                _Journal(self).recover()
            if set(self.project)!={'schema','id','name','origin'} or self.project['schema']!='optimizer-project/1':
                raise _ServiceError('checkpoint_corrupt','Unknown project schema/fields.')
            required={'schema','project_id','generation','contexts','selected','bindings','objects','bookmarks','pins','ancestry','cursor','redo','policy','environment'}
            if set(self.index) not in (required,required|{'jobs'}) or self.index['schema']!='optimizer-session/1' or self.index['project_id']!=self.project['id']:
                raise _ServiceError('checkpoint_corrupt','Unknown session schema/identity/fields.')
            self.policy(**self.index['policy'])
            if type(self.index['generation']) is not int or self.index['generation']<0:
                raise _ServiceError('checkpoint_corrupt','Invalid store generation.')
            fields={'hash','blobs','available','resume_level','type','inputs','context','configuration'}
            for oid,entry in self.index['objects'].items():
                if not isinstance(oid,str) or not _ID.fullmatch(oid) or set(entry)!=fields or type(entry['available']) is not bool:
                    raise _ServiceError('checkpoint_corrupt','Invalid checkpoint index entry.')
            jobs=self.index.get('jobs')
            if jobs is not None:
                if not isinstance(jobs,dict) or set(jobs)!={'schema','runtime_id','config','signature','records'} or jobs['schema']!='optimizer-jobs/1' or not isinstance(jobs['records'],dict):
                    raise _ServiceError('checkpoint_corrupt','Invalid durable job registry.')
                for job,entry in jobs['records'].items():
                    if not isinstance(job,str) or not _ID.fullmatch(job) or not isinstance(entry,dict) or set(entry) not in ({'payload','reserved_bytes'},{'payload','reserved_bytes','status'},{'payload','reserved_bytes','status','handles'}) or type(entry['reserved_bytes']) is not int or entry['reserved_bytes']<0 or entry['payload'] not in self.index['objects']:
                        raise _ServiceError('checkpoint_corrupt','Invalid job payload/reservation.')
                    handles=entry.get('handles',[])
                    if not isinstance(handles,list) or any(not isinstance(oid,str) or oid not in self.index['objects'] or self.index['objects'][oid]['type']!='JobHandle' for oid in handles):
                        raise _ServiceError('checkpoint_corrupt','Invalid durable job handle references.')
            protected=self.protected()
            if any(not self.index['objects'].get(oid,{}).get('available') for oid in protected):
                raise _ServiceError('checkpoint_corrupt','Active/pinned payload is missing from index.')
            self.recover()
        except BaseException:
            self.lock.close(); raise

    @classmethod
    def create(cls, name, root='.', *, max_bytes=536870912, keep_recent=20, undo_depth=10, origin=None):
        if not isinstance(name,str) or not _NAME.fullmatch(name):
            raise _ServiceError('invalid_arguments','Experiment name must be 1–64 safe letters/numbers/hyphens/underscores.')
        policy=cls.policy(max_bytes,keep_recent,undo_depth)
        parent=Path(root).expanduser().resolve()/'.optimizer'
        parent.mkdir(parents=True,exist_ok=True)
        target=parent/name
        if target.exists(): raise _ServiceError('project_exists','Experiment already exists; open it or choose a new name.')
        temporary=Path(tempfile.mkdtemp(prefix='.initialize-',dir=parent))
        try:
            for folder in ('runtime','inputs','checkpoints/manifests','checkpoints/blobs'):
                (temporary/folder).mkdir(parents=True,exist_ok=True)
            identity=uuid4().hex
            project={'schema':'optimizer-project/1','id':identity,'name':name,'origin':origin}
            index={'schema':'optimizer-session/1','project_id':identity,'generation':0,'contexts':{},'selected':None,
                   'bindings':{},'objects':{},'bookmarks':{},'pins':[],'ancestry':{},'cursor':{},'redo':{},
                   'policy':policy,'environment':_environment()}
            _atomic(temporary/'project.json',_json(project)); _atomic(temporary/'session.json',_json(index))
            os.rename(temporary,target)
            return cls(target)
        finally:
            if temporary.exists(): shutil.rmtree(temporary)

    @staticmethod
    def policy(max_bytes,keep_recent,undo_depth):
        if type(max_bytes) is not int or max_bytes<1 or any(type(v) is not int or v<0 for v in (keep_recent,undo_depth)):
            raise _ServiceError('invalid_arguments','Storage bytes must be positive; recent/undo counts nonnegative integers.')
        return dict(max_bytes=max_bytes,keep_recent=keep_recent,undo_depth=undo_depth)

    def close(self):
        if not self.lock.closed: self.lock.close()

    def _path(self, relative):
        target=self.path/relative
        if any(p.is_symlink() for p in (target,*target.parents) if p!=self.path and p.is_relative_to(self.path)) or not target.resolve().is_relative_to(self.path):
            raise _ServiceError('checkpoint_corrupt','Store path escapes experiment or is symlinked.')
        return target

    def read(self, relative, limit=64*1024*1024):
        target=self._path(relative)
        try:
            with target.open('rb') as handle: payload=handle.read(limit+1)
        except FileNotFoundError as exc:
            raise _ServiceError('checkpoint_missing',f'Missing {relative}.') from exc
        if len(payload)>limit: raise _ServiceError('codec_limit','Stored file exceeds read budget.')
        return payload

    def read_json(self,relative):
        import json
        try: return json.loads(self.read(relative), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (ValueError, RecursionError) as exc: raise _ServiceError('checkpoint_corrupt','Invalid stored JSON.') from exc

    def manifest(self,identity):
        if not isinstance(identity,str) or not _ID.fullmatch(identity):
            raise _ServiceError('checkpoint_corrupt','Invalid checkpoint identity.')
        entry=self.index['objects'].get(identity)
        if entry is None: raise _ServiceError('checkpoint_missing','Unknown checkpoint.')
        if not entry['available']: raise _ServiceError('payload_pruned','Checkpoint payload has been pruned.')
        raw=self.read(f'checkpoints/manifests/{identity}.json')
        if _hash(raw)!=entry['hash']: raise _ServiceError('checkpoint_corrupt','Manifest hash mismatch.')
        import json
        manifest=json.loads(raw)
        if set(manifest)!={'schema','payload'} or manifest['schema']!='optimizer-checkpoint/1':
            raise _ServiceError('checkpoint_corrupt','Unknown checkpoint schema/fields.')
        return manifest['payload']

    def codec(self,max_bytes):
        def read_blob(digest):
            if not isinstance(digest,str) or len(digest)!=64 or not _ID.fullmatch(digest):
                raise _ServiceError('checkpoint_corrupt','Invalid blob reference.')
            return self.read(f'checkpoints/blobs/{digest}.npz', max_bytes+8192)
        return _Codec(read_blob,max_bytes=max_bytes)

    def decode(self,identity,max_bytes):
        import zipfile
        try:
            data=self.codec(max_bytes).decode(self.manifest(identity))
            fields={'value','context','configuration','method_key','recipe','resume_level','environment','inputs'}
            if not isinstance(data,dict) or set(data)!=fields or data['resume_level'] not in {'exact','controls_only','variables_only','analysis_only'}:
                raise _ServiceError('checkpoint_corrupt','Unknown checkpoint payload fields/resume level.')
            return data
        except (ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            raise _ServiceError("checkpoint_corrupt", "Invalid typed checkpoint payload.") from exc

    def _files(self):
        return [p for p in self.path.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(self.path).parts]

    def usage(self):
        return sum(p.stat().st_size for p in self._files())

    def protected(self,index=None):
        index=self.index if index is None else index
        records=index.get('jobs',{}).get('records',{}).values()
        return set(index['pins'])|{c['head'] for c in index['contexts'].values() if c['head']}|{
            oid for record in records for oid in [record['payload'],*record.get('handles',[])]}

    def reclaim(self, targets, index=None):
        index=self.index if index is None else index
        targets=set(targets)
        removed={blob for oid in targets for blob in index['objects'][oid]['blobs']}
        kept={blob for oid,e in index['objects'].items() if e['available'] and oid not in targets for blob in e['blobs']}
        kept.update(self._recipe_blobs(index))
        exclusive=removed-kept
        paths=[self._path(f'checkpoints/manifests/{oid}.json') for oid in targets]
        paths += [self._path(f'checkpoints/blobs/{blob}.npz') for blob in exclusive]
        return {'reclaimable_bytes':sum(p.stat().st_size for p in paths if p.exists()),
                'removed_blobs':sorted(exclusive),'shared_blobs_kept':sorted(removed & kept)}

    def recover(self):
        # OS leases die with the process; no PID guessing or worker termination.
        if self.read_only: return
        for p in self.path.rglob('.prepare-*'):
            if p.is_file() and not p.is_symlink(): p.unlink()
        known={f'{oid}.json' for oid,e in self.index['objects'].items() if e['available']}
        for p in (self.path/'checkpoints/manifests').glob('*.json'):
            if p.name not in known and not p.is_symlink(): p.unlink()
        self._collect_blobs()
        self._collect_inputs()

    @staticmethod
    def _recipe_blobs(index):
        live=set()
        def arrays(value):
            if isinstance(value,dict):
                if '$array' in value: live.add(value['$array'])
                for child in value.values(): arrays(child)
            elif isinstance(value,list):
                for child in value: arrays(child)
        arrays(index['contexts']); arrays(index['ancestry'])
        return live

    def _collect_blobs(self):
        live=self._recipe_blobs(self.index)
        for oid,entry in self.index['objects'].items():
            if entry['available']: live.update(entry['blobs'])
        for p in (self.path/'checkpoints/blobs').glob('*.npz'):
            if p.stem not in live and not p.is_symlink(): p.unlink()

    def _collect_inputs(self):
        live=set()
        def digests(value):
            if isinstance(value,str) and len(value)==64 and _ID.fullmatch(value): live.add(value)
            elif isinstance(value,dict):
                for child in value.values(): digests(child)
            elif isinstance(value,list):
                for child in value: digests(child)
        digests(self.index)
        for oid,entry in self.index['objects'].items():
            if entry['available']: digests(self.manifest(oid))
        for path in (self.path/'inputs').iterdir():
            if path.is_dir() and not path.is_symlink() and len(path.name)==64 and _ID.fullmatch(path.name) and path.name not in live:
                shutil.rmtree(path)

    def commit(self,index,files,guard=None,*,history=None):
        if self.read_only: raise _ServiceError('read_only','Experiment is open read-only.')
        if self._path('history.jsonl').exists() and history is None:
            raise _ServiceError('history_owned','This experiment requires a recorded frontend transaction; open read-only or fork an index-owned experiment.')
        self.index_check=index['generation']==self.index['generation']+1
        if not self.index_check: raise _ServiceError('revision_conflict','Store generation changed.')
        # Count all physical preparation bytes including the replacement index.
        outcome=None
        if history is not None:
            journal,data,command_id,session_id=history
            outcome={**data,'index':index}
        extra=sum(len(data) for relative,data in files.items() if not self._path(relative).exists())+len(_json(index))
        if outcome is not None: extra+=len(_json(outcome))+1024
        reserved=sum(record['reserved_bytes'] for record in index.get('jobs',{}).get('records',{}).values())
        if self.usage()+extra+reserved > index['policy']['max_bytes']:
            raise _ServiceError('storage_full','Insufficient preparation capacity; prune/unpin or raise the explicit byte limit.')
        created=[]
        committed=False
        try:
            for relative,data in files.items():
                target=self._path(relative)
                if not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    created.append(target); _atomic(target,data)
                elif target.read_bytes()!=data:
                    raise _ServiceError('checkpoint_corrupt','Immutable payload collision.')
            if guard is not None: guard()
            self.warning=None
            if outcome is not None:
                journal.outcome(outcome,command_id,session_id)
                # The durable outcome precedes the replaceable session index.
                committed=True
                self.index=index
            try: _atomic(self.path/'session.json',_json(index))
            except _CommittedWriteError as exc:
                self.warning='Index committed; directory fsync failed: '+str(exc)
            except (OSError,KeyboardInterrupt) as exc:
                if not committed: raise
                self.warning='History committed; index recovery required: '+str(exc)
            committed=True
            self.index=index
        finally:
            if not committed:
                for target in created:
                    if target.exists(): target.unlink()
        # After index commit, cleanup failure cannot turn success into rollback.
        try:
            for oid,entry in index['objects'].items():
                if not entry['available']:
                    self._path(f'checkpoints/manifests/{oid}.json').unlink(missing_ok=True)
            self._collect_blobs()
            self._collect_inputs()
        except (OSError,_ServiceError,KeyboardInterrupt) as exc:
            self.warning=(self.warning+"; " if self.warning else "")+"Committed; cleanup deferred: "+str(exc)

    def clone(self,name,root=None):
        destination=self.create(name, str(self.path.parent.parent) if root is None else root,
                                **self.index['policy'],origin={'project_id':self.project['id'],'generation':self.index['generation']})
        try:
            index=copy.deepcopy(self.index); index['project_id']=destination.project['id']; index['generation']=1
            # A fork inherits numerical artifacts, never live coordinator jobs.
            index.pop('jobs',None)
            files={p.relative_to(self.path).as_posix():p.read_bytes() for p in self._files()
                   if p.relative_to(self.path).parts[0] in ('checkpoints','inputs')}
            destination.commit(index,files)
            return destination
        except BaseException:
            destination.close(); shutil.rmtree(destination.path); raise
