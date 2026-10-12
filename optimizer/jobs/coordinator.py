"""Authenticated local socket transport; the owner alone holds the store lease."""
from __future__ import annotations
import collections
import json
import os
from pathlib import Path
import secrets
import socket
import struct
import subprocess
import shutil
import sys
import tempfile
import time
from uuid import uuid4

from optimizer.operations.contracts import OperationResponse, ObjectRef, _ServiceError
from optimizer.persistence.store import _environment
from optimizer.cli.language import _Terminal
from optimizer.cli.input import _MAX_BYTES

_LIMIT=8*1024*1024

def _recv(connection):
    def exact(count):
        parts=[]
        while count:
            part=connection.recv(count)
            if not part: raise ConnectionError('Incomplete coordinator message.')
            parts.append(part); count-=len(part)
        return b''.join(parts)
    size=struct.unpack('!I',exact(4))[0]
    if size>_LIMIT: raise _ServiceError('input_limit','Coordinator message exceeds 8 MiB.')
    value=json.loads(exact(size),parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    if not isinstance(value,dict): raise ValueError('Coordinator message must be an object.')
    return value

def _send(connection,value):
    from optimizer.checkpoints.codec import _json
    raw=_json(value)
    if len(raw)>_LIMIT: raise _ServiceError('input_limit','Coordinator response exceeds 8 MiB.')
    connection.sendall(struct.pack('!I',len(raw))+raw)

def _identity(path):
    path=Path(path).expanduser().resolve(strict=True)
    metadata=path/'runtime/coordinator.json'
    if metadata.is_symlink() or metadata.stat().st_size>8192:
        raise _ServiceError('coordinator_identity','Invalid coordinator identity file.')
    if metadata.stat().st_uid!=os.getuid() or metadata.stat().st_mode & 0o077:
        raise _ServiceError('coordinator_identity','Coordinator credentials must be private to the current user.')
    value=json.loads(metadata.read_bytes())
    if set(value)!={'schema','project_id','identity','socket','token','environment'} or value['schema']!='optimizer-coordinator/1':
        raise _ServiceError('coordinator_identity','Unknown coordinator identity.')
    if not isinstance(value['socket'],str) or not isinstance(value['token'],str) or len(value['token'])!=64:
        raise _ServiceError('coordinator_identity','Invalid coordinator address/credential.')
    endpoint=Path(value['socket'])
    if endpoint.is_symlink() or endpoint.parent.is_symlink() or not endpoint.parent.name.startswith('opt-coord-') or not endpoint.parent.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise _ServiceError('coordinator_identity','Invalid local socket owner path.')
    if endpoint.parent.stat().st_uid!=os.getuid() or endpoint.parent.stat().st_mode & 0o077:
        raise _ServiceError('coordinator_identity','Local socket directory must be private.')
    if value['project_id']!=json.loads((path/'project.json').read_bytes())['id']:
        raise _ServiceError('coordinator_identity','Coordinator belongs to another experiment.')
    if value['environment']!=_environment(): raise _ServiceError('environment_changed','Coordinator/package environment differs; explicitly shut down/recover with the matching version.')
    return path,value

def request(path,method,**arguments):
    try:
        path,identity=_identity(path)
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
            connection.settimeout(3); connection.connect(identity['socket'])
            connection.settimeout(45)
            _send(connection,{'token':identity['token'],'identity':identity['identity'],
                              'request_id':uuid4().hex,'method':method,**arguments})
            response=_recv(connection)
    except _ServiceError: raise
    except (OSError,ValueError,ConnectionError) as exc:
        raise _ServiceError('coordinator_unavailable','Coordinator unavailable; no command is automatically retried. Inspect history before repeating uncertain work.') from exc
    if response.get('error'):
        raise _ServiceError(response['error']['code'],response['error']['message'])
    return response

def start(path,mode='exact'):
    path=Path(path).expanduser().resolve(strict=True)
    try:
        request(path,'ping')
        return path
    except _ServiceError: pass
    # A separate bootstrap lease prevents racing launches; the writer lease is
    # still the final authority. Never terminate a PID from stale metadata.
    import fcntl
    lock_path=path/'runtime/coordinator-start.lock'
    if lock_path.is_symlink(): raise _ServiceError('coordinator_identity','Symlinked coordinator start lease.')
    with lock_path.open('a+b') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc: raise _ServiceError('project_locked','Another coordinator launch is in progress.') from exc
        environment=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[2]),PYTHONDONTWRITEBYTECODE='1')
        process=subprocess.Popen([sys.executable,'-B','-m','optimizer.jobs.coordinator',str(path),mode],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            start_new_session=True,env=environment)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if process.poll() is not None:
                raise _ServiceError('coordinator_start_failed','Coordinator could not acquire/open the experiment; check its writer lease, source/environment and explicit restore mode.')
            try:
                request(path,'ping'); return path
            except _ServiceError: time.sleep(.05)
        # This is the exact child just launched, never a metadata PID.
        process.terminate(); process.wait(timeout=3)
        raise _ServiceError('coordinator_start_failed','Coordinator startup timed out.')

class _RemoteTerminal(_Terminal):
    def __init__(self,path,**kwargs):
        super().__init__(**kwargs)
        self.path=Path(path).expanduser().resolve(strict=True)
        request(self.path,'ping')

    def submit(self,source):
        try:
            if self.progress is None:
                reply=request(self.path,'submit',source=source,cwd=os.getcwd())
            else:
                with self.progress('coordinator'):
                    reply=request(self.path,'submit',source=source,cwd=os.getcwd())
            self.timings.update(reply.get('timings',{}))
            for row in reply['responses']:
                refs=tuple(ObjectRef(r['object_id'],tuple(r['path'])) for r in row.pop('object_refs'))
                self.emit(OperationResponse(**row,object_refs=refs))
            self.closed=reply['closed']
            return reply['status']
        except _ServiceError as exc:
            self.emit(OperationResponse('coordinator-client',None,'error',error={'code':exc.code,'message':str(exc)}))
            return 2

    def completions(self):
        return request(self.path,'completions')['names']

    def cancel_input(self,source):
        return request(self.path,'input_cancelled',source=source)['status']

    def finish(self,reason='frontend_close'):
        if self.closed: return
        try: request(self.path,'detach',origin=reason)
        except _ServiceError as exc:
            self.emit(OperationResponse('coordinator-detach',None,'error',error={'code':exc.code,'message':str(exc)}))
        self.closed=True

def serve(path,mode):
    from optimizer.jobs.durable import recover, flush
    from optimizer.persistence.store import _atomic
    from optimizer.checkpoints.codec import _json
    from optimizer.operations import OperationRequest
    terminal=_Terminal(); service=terminal.service
    from optimizer.operations import OperationRequest
    def open_store(staged):
        operation=OperationRequest('project.open',{'path':str(path),'mode':mode})
        response=staged.execute(operation); terminal.history.capture(response,operation)
        return 0 if response.ok else 2
    code=terminal.history.run('',None,open_store,origin='coordinator_attach')
    if code: raise _ServiceError('coordinator_start_failed','Experiment open failed.')
    store=service._persistence.store
    stale=store._path('runtime/coordinator.json')
    if stale.exists() and not stale.is_symlink() and stale.stat().st_size<=8192:
        try:
            old=json.loads(stale.read_bytes()); endpoint=Path(old['socket'])
            if old['project_id']==store.project['id'] and endpoint.name=='socket' and endpoint.parent.name.startswith('opt-coord-') and not endpoint.parent.is_symlink() and endpoint.parent.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()) and endpoint.parent.stat().st_uid==os.getuid() and not endpoint.parent.stat().st_mode & 0o077:
                endpoint.unlink(missing_ok=True); endpoint.parent.rmdir()
        except (OSError,ValueError,TypeError,KeyError): pass
    manager=service._jobs
    manager.durable=True
    manager.runtime_id=store.project['id']
    transport=Path(tempfile.mkdtemp(prefix='opt-coord-'))
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    try:
        listener.bind(str(transport/'socket')); listener.listen(8); listener.settimeout(.1)
    except BaseException:
        listener.close(); shutil.rmtree(transport,ignore_errors=True); manager.durable=False; service.close(); raise
    metadata=store._path('runtime/coordinator.json')
    identity={'schema':'optimizer-coordinator/1','project_id':store.project['id'],'identity':uuid4().hex,
              'socket':str(transport/'socket'),'token':secrets.token_hex(32),'environment':_environment()}
    cache=collections.OrderedDict(); cache_bytes=0
    try:
        recover(manager,store)
        flush(terminal)
        _atomic(metadata,_json(identity)); metadata.chmod(0o600)
        while not manager.shutdown_requested:
            publication_error=None
            try: flush(terminal)
            except (OSError,_ServiceError) as exc: publication_error=exc
            try: connection,_=listener.accept()
            except socket.timeout: continue
            with connection:
                connection.settimeout(3)
                try:
                    incoming=_recv(connection)
                    if set(incoming)-{'token','identity','request_id','method','source','cwd','origin'}:
                        raise _ServiceError('invalid_arguments','Unknown coordinator request fields.')
                    if not secrets.compare_digest(str(incoming.get('token','')),identity['token']) or incoming.get('identity')!=identity['identity']:
                        raise _ServiceError('coordinator_auth','Coordinator authentication failed.')
                    rid=incoming.get('request_id')
                    if not isinstance(rid,str) or len(rid)!=32: raise _ServiceError('invalid_arguments','Request needs a UUID identity.')
                    if rid in cache:
                        _send(connection,cache[rid]); continue
                    method=incoming.get('method')
                    manager.client_detach=False; terminal.closed=False
                    if method=='ping': reply={'identity':identity['identity'],'lifetime':'coordinator'}
                    elif method=='completions': reply={'names':terminal.completions()}
                    elif method=='detach':
                        origin=incoming.get('origin','frontend_close')
                        if origin not in {'frontend_close','input_cancelled'}: raise _ServiceError('invalid_arguments','Unknown detach origin.')
                        def detach(staged):
                            operation=OperationRequest('jobs.detach')
                            response=staged.execute(operation); terminal.history.capture(response,operation)
                            return 0 if response.ok else 2
                        reply={'status':terminal.history.run('',None,detach,origin=origin)}
                    elif method=='input_cancelled':
                        source=incoming.get('source')
                        if not isinstance(source,str) or len(source.encode())>_MAX_BYTES: raise _ServiceError('input_limit','Cancelled input exceeds its budget.')
                        reply={'status':terminal.history.run(source,None,lambda staged:130,origin='input_cancelled')}
                    elif method=='submit':
                        source=incoming.get('source')
                        if not isinstance(source,str) or len(source.encode())>_MAX_BYTES:
                            raise _ServiceError('input_limit','Command text exceeds terminal input budget.')
                        # A full store can be repaired by the existing explicit
                        # policy-increase statement, without bypassing history.
                        if publication_error and terminal.history._capacity_override(source,store) is None:
                            raise _ServiceError('coordinator_publication_failed','Pending outcomes are unacknowledged; repair storage explicitly before retrieving or submitting work.')
                        original=Path.cwd(); target=Path(incoming.get('cwd',str(original))).resolve(strict=True)
                        responses=[]; terminal.emit=lambda response:responses.append(response.to_dict())
                        try:
                            os.chdir(target); status=terminal.submit(source)
                        finally: os.chdir(original)
                        flush(terminal)
                        reply={'status':status,'responses':responses,'closed':terminal.closed or manager.client_detach or manager.shutdown_requested,
                               'timings':terminal.timings}
                    else: raise _ServiceError('invalid_arguments','Unknown coordinator request.')
                    size=len(_json(reply)); cache[rid]=reply; cache_bytes+=size
                    while len(cache)>128 or cache_bytes>16*1024*1024:
                        _,old=cache.popitem(last=False); cache_bytes-=len(_json(old))
                    _send(connection,reply)
                except (OSError,ValueError,TypeError,KeyError,_ServiceError) as exc:
                    try: _send(connection,{'error':{'code':getattr(exc,'code','coordinator_request_failed'),'message':str(exc)}})
                    except OSError: pass
    finally:
        manager.close()
        try: flush(terminal)
        except (OSError,_ServiceError): pass
        if metadata.exists(): metadata.unlink()
        listener.close(); shutil.rmtree(transport,ignore_errors=True); manager.durable=False; service.close()

if __name__=='__main__':
    serve(Path(sys.argv[1]).resolve(strict=True),sys.argv[2])
