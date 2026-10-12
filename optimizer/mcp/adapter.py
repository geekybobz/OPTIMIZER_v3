"""One MCP connection selects one existing experiment coordinator."""
from __future__ import annotations
import os
from pathlib import Path
from optimizer.jobs.coordinator import request, start
from optimizer.operations.contracts import _ServiceError
from optimizer.operations.frontend import _bounded

class _Adapter:
    def __init__(self):
        self.path=None

    def attach(self,path,*,start_coordinator=False,restore_mode='exact'):
        if self.path is not None: raise _ServiceError('project_open','Detach before attaching another experiment.')
        if not isinstance(path,str) or type(start_coordinator) is not bool or restore_mode not in {'exact','controls_only','variables_only'}:
            raise _ServiceError('invalid_arguments','Supply a path, explicit startup flag and supported restore mode.')
        target=Path(path).expanduser().resolve(strict=True)
        if start_coordinator: start(target,restore_mode)
        result=request(target,'native',action='attach',payload={},cwd=os.getcwd())
        if result['status']=='ok': self.path=target
        return result

    def call(self,action,payload):
        if self.path is None: raise _ServiceError('missing_project','Attach an experiment first.')
        _bounded(payload)
        result=request(self.path,'native',action=action,payload=payload,cwd=os.getcwd())
        if action=='execute' and result['status']=='ok' and payload.get('operation') in {'jobs.shutdown','jobs.detach'}:
            self.path=None
        return result

    def detach(self):
        if self.path is None: return {'status':'ok','metadata':{'summary':{'attached':False}}}
        result=self.call('detach',{})
        if result['status']=='ok': self.path=None
        return result

    def close(self):
        if self.path is not None:
            try: self.detach()
            except (OSError,_ServiceError): self.path=None
