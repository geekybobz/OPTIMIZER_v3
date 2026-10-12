"""Versioned, hash-chained statement history; outcomes own durable commits."""
from __future__ import annotations
import copy
import json
import os
from collections import deque
from datetime import datetime, timezone
from uuid import uuid4
from optimizer.checkpoints.codec import _json, _hash
from optimizer.operations.contracts import _ServiceError

_SCHEMA = 'optimizer-history/1'
_MAX_LINE = 32*1024*1024
_RESERVE = 131072

class _Journal:
    def __init__(self, store):
        self.store = store
        self.path = store._path('history.jsonl')
        self.sequence = 0
        self.digest = None
        self.latest_index = None
        self.pending = {}
        for row in self.rows():
            self.sequence, self.digest = row['sequence'], row['hash']
            if row['kind'] == 'baseline': self.latest_index = row['data']['index']
            elif row['kind'] == 'submitted': self.pending[row['command_id']] = row
            elif row['kind'] == 'outcome':
                self.pending.pop(row['command_id'], None)
                if row['data'].get('index') is not None: self.latest_index = row['data']['index']

    def rows(self):
        if not self.path.exists(): return
        previous, sequence, pending = None, 0, {}
        with self.path.open('rb') as handle:
            while True:
                raw = handle.readline(_MAX_LINE+1)
                if not raw: break
                if len(raw)>_MAX_LINE or not raw.endswith(b'\n'):
                    raise _ServiceError('history_corrupt','Truncated or oversized history record; preserve the file for explicit repair.')
                try:
                    row=json.loads(raw, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
                    digest=row.pop('hash')
                    fields={'schema','sequence','previous','kind','command_id','session_id','timestamp','data'}
                    if set(row)!=fields or row['schema']!=_SCHEMA or type(row['sequence']) is not int or row['sequence']!=sequence+1 or row['previous']!=previous or _hash(_json(row))!=digest:
                        raise ValueError('Invalid schema, sequence or hash chain.')
                    kind, cid=row['kind'],row['command_id']
                    if not isinstance(row['data'],dict): raise ValueError('Invalid record data.')
                    if not isinstance(row['timestamp'],str):raise ValueError('Invalid timestamp.')
                    if kind=='baseline':
                        if sequence!=0 or cid is not None or row['session_id'] is not None or set(row['data'])!={'index','origin'} or not isinstance(row['data']['index'],dict): raise ValueError('Invalid baseline.')
                    elif kind=='submitted':
                        if sequence==0 or pending or not isinstance(cid,str) or not cid or not isinstance(row['session_id'],str):raise ValueError('Invalid submission.')
                        data=row['data']
                        if set(data)!={'raw','expression','state_before','environment','origin'} or not isinstance(data['raw'],str) or not isinstance(data['environment'],dict) or not isinstance(data['state_before'],dict):raise ValueError('Invalid submission fields.')
                        pending[cid]=row['session_id']
                    elif kind=='outcome':
                        data=row['data']
                        if cid not in pending or pending[cid]!=row['session_id'] or set(data)!={'status','state_before','state_after','operations','error','summary','index'} or data['status'] not in {'ok','error','interrupted'} or not isinstance(data['operations'],list):raise ValueError('Unpaired or invalid outcome.')
                        if data['index'] is not None and (data['status']!='ok' or not isinstance(data['index'],dict)):raise ValueError('Invalid committed index.')
                        pending.pop(cid)
                    else:raise ValueError('Unknown mandatory record kind.')
                    row['hash']=digest
                except (ValueError, TypeError, KeyError, RecursionError) as exc:
                    raise _ServiceError('history_corrupt','Invalid command history: '+str(exc)) from exc
                previous,sequence=digest,row['sequence']
                yield row

    def append(self, kind, data, *, command_id=None, session_id=None, reserve=0):
        row=dict(schema=_SCHEMA,sequence=self.sequence+1,previous=self.digest,kind=kind,
                 command_id=command_id,session_id=session_id,timestamp=datetime.now(timezone.utc).isoformat(),data=data)
        digest=_hash(_json(row)); row['hash']=digest; raw=_json(row)+b'\n'
        if len(raw)>_MAX_LINE:raise _ServiceError('history_limit','History record exceeds its line budget.')
        limit=getattr(self,'capacity_override',self.store.index['policy']['max_bytes'])
        if self.store.usage()+len(raw)+reserve>limit:
            raise _ServiceError('storage_full','Command history capacity exhausted; raise the experiment byte limit explicitly.')
        existed=self.path.exists()
        offset=self.path.stat().st_size if existed else 0
        # An unsuccessful append never becomes an acknowledged commit.
        try:
            with self.path.open('ab') as handle:
                handle.write(raw);handle.flush();os.fsync(handle.fileno())
            if not existed:
                directory=os.open(self.path.parent,os.O_RDONLY)
                try:os.fsync(directory)
                finally:os.close(directory)
        except BaseException:
            if self.path.exists():
                if not existed:self.path.unlink()
                else:
                    with self.path.open('r+b') as handle:
                        handle.truncate(offset);handle.flush();os.fsync(handle.fileno())
            raise
        self.sequence,self.digest=row['sequence'],digest
        if kind=='baseline' or (kind=='outcome' and data.get('index') is not None):self.latest_index=data['index']
        if kind=='submitted':self.pending[command_id]=row
        elif kind=='outcome':self.pending.pop(command_id,None)
        return row

    def begin(self, data, command_id, session_id):
        return self.append('submitted',data,command_id=command_id,session_id=session_id,reserve=_RESERVE)

    def outcome(self,data,command_id,session_id):
        return self.append('outcome',data,command_id=command_id,session_id=session_id)

    def recover(self):
        index=self.latest_index
        if index is None or index.get('project_id')!=self.store.project['id'] or index.get('schema')!='optimizer-session/1':
            raise _ServiceError('history_corrupt','History has no valid experiment baseline.')
        disk=self.store.index
        if type(disk.get('generation')) is not int:
            raise _ServiceError('history_corrupt','Invalid session generation.')
        if disk['generation']>index['generation'] or (disk['generation']==index['generation'] and disk!=index):
            raise _ServiceError('unrecorded_state','Session index contains an unrecorded change; automatic recovery refused.')
        self.store.index=copy.deepcopy(index)
        if self.store.read_only:return
        from optimizer.persistence.store import _atomic
        if disk!=index:_atomic(self.store.path/'session.json',_json(index))
        for cid,row in tuple(self.pending.items()):
            self.outcome(dict(status='interrupted',error={'code':'interrupted','message':'Process ended before a durable outcome.'},
                              state_before=row['data'].get('state_before'),state_after=None,operations=[],summary=None,index=None),cid,row['session_id'])

    def commands(self,limit=50,before=None):
        if type(limit) is not int or not 1<=limit<=200:raise _ServiceError('invalid_arguments','history limit must be an integer from 1 to 200.')
        submitted={};result=deque(maxlen=limit);found=before is None
        for row in self.rows():
            if before is not None and row['command_id']==before and row['kind']=='submitted':found=True;break
            if row['kind']=='submitted':submitted[row['command_id']]=row
            elif row['kind']=='outcome':
                intent=submitted.pop(row['command_id'])
                result.append(dict(command_id=row['command_id'],session_id=row['session_id'],sequence=row['sequence'],
                                   raw=intent['data']['raw'],expression=intent['data'].get('expression'),
                                   status=row['data']['status'],summary=row['data'].get('summary'),error=row['data'].get('error')))
        if not found:raise _ServiceError('history_missing','Unknown history command.')
        return list(result)
