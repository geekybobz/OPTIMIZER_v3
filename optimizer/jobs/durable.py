"""Job records use the experiment's existing immutable checkpoint store."""
from __future__ import annotations
import dataclasses
import json
from optimizer.checkpoints.codec import _Codec, _json, _hash
from optimizer.operations.contracts import _ServiceError
from optimizer.session import _owned, _compatible, _size
from optimizer.jobs.manager import _Job, _TERMINAL
from optimizer.jobs.contracts import JobsConfig
from optimizer.persistence.store import _environment

_FIELDS = ('handle','packet','sources','source_path','recipe','layout','family','input_refs',
           'worker_bytes','result_bytes','cache_bytes','scratch_bytes','status','error',
           'cancel_requested','forced','cancellation_boundary','pause_state','environment','parent','result')

def _signature(config, records):
    return _hash(_json({'config':dataclasses.asdict(config), 'records':[
        [identity, record['status'],record['cancel_requested'],record['error'],
         record['pause_state'] is not None,record['result'] is not None]
        for identity,record in records.items()]}))

def _snapshot(manager, actions=()):
    with manager._condition:
        records={key:{field:_owned(getattr(job,field)) for field in _FIELDS}
                 for key,job in manager.records.items()}
        observed={key:record['status'] for key,record in records.items()}
        config=manager.config
        for kind,value in actions:
            if kind=='configure': config=value
            elif kind=='submit': records[value.handle.job_id]={field:_owned(getattr(value,field)) for field in _FIELDS}
            elif kind=='release': records.pop(value,None)
            elif kind in {'cancel','pause','resume'} and value in records:
                record=records[value]
                if kind=='cancel' and record['status'] not in _TERMINAL:
                    record['cancel_requested']=True
                    if manager.records[value].process is None: record['status']='cancelled'
                elif kind=='pause' and record['status']=='queued': record['status']='paused'
                elif kind=='pause' and record['status']=='running': record['status']='pausing'
                elif kind=='resume': record['status']='queued' if manager.records[value].process is None else 'running'
        return config, records, observed

def signature(manager):
    with manager._condition:
        records={key:{field:getattr(job,field) for field in _FIELDS} for key,job in manager.records.items()}
        return _signature(manager.config,records)

def persist(manager, persistence, session):
    if not manager.durable: return
    config,records,observed=_snapshot(manager,persistence.service._job_actions)
    index=persistence.index
    previous=index.get('jobs',{}).get('records',{})
    history=persistence.service._history
    if history and history.command and history.command['origin']!='coordinator':
        changes=[{'job_id':key,'before':previous[key].get('status'),'status':status}
                 for key,status in observed.items() if key in previous and previous[key].get('status')!=status]
        if changes:
            history.events.append({'operation':'jobs.transition','origin':'coordinator','arguments':{'transitions':changes},
                'effective_arguments':{},'object_refs':[],'context':None,'bind':None,'activate':None,
                'status':'ok','error':None,'summary':{'transitions':changes}})
    registry={'schema':'optimizer-jobs/1','runtime_id':manager.runtime_id,
              'config':dataclasses.asdict(config),'signature':_signature(config,records),'records':{}}
    for identity,record in records.items():
        codec=_Codec(max_bytes=persistence.service.session.max_bytes)
        envelope={'value':record,'context':None,'configuration':None,'method_key':None,'recipe':None,
                  'resume_level':'analysis_only','environment':record['environment'],'inputs':()}
        raw=_json({'schema':'optimizer-checkpoint/1','payload':codec.encode(envelope)})
        digest=_hash(raw)
        persistence.files[f'checkpoints/manifests/{digest}.json']=raw
        persistence.files.update({f'checkpoints/blobs/{key}.npz':data for key,data in codec.blobs.items()})
        index['objects'][digest]={'hash':digest,'blobs':list(codec.blobs),'available':True,'resume_level':'analysis_only',
                                 'type':'JobRecord','inputs':[],'context':None,'configuration':None}
        handles=[oid for oid,stored in session._objects.items()
                 if type(stored.value) is type(record['handle']) and stored.value==record['handle']]
        registry['records'][identity]={'payload':digest,'status':record['status'],'handles':handles,'reserved_bytes':
            8*record['result_bytes']+65536 if record['status'] not in _TERMINAL else 0}
    index['jobs']=registry

def recover(manager, store):
    registry=store.index.get('jobs')
    if registry is None: return
    if registry.get('schema')!='optimizer-jobs/1' or set(registry)!={'schema','runtime_id','config','signature','records'}:
        raise _ServiceError('checkpoint_corrupt','Unknown durable job registry.')
    config=JobsConfig(**registry['config'])
    if len(registry['records'])>config.max_jobs: raise _ServiceError('job_quota','Saved jobs exceed the configured count.')
    records={}
    for identity,entry in registry['records'].items():
        record=store.decode(entry['payload'],config.coordinator_bytes)['value']
        if set(record)!=set(_FIELDS) or record['handle'].job_id!=identity or record['handle'].runtime_id!=registry['runtime_id']:
            raise _ServiceError('checkpoint_corrupt','Durable job identity/fields differ.')
        if record['status'] not in _TERMINAL|{'queued','running','pausing','paused'}:
            raise _ServiceError('checkpoint_corrupt','Unknown durable job status.')
        codec=_Codec(max_bytes=config.input_bytes)
        manifest=_json(codec.encode(record['packet']))
        if len(manifest)+sum(map(len,codec.blobs.values()))+sum(map(len,record['sources'].values()))>config.input_bytes:
            raise _ServiceError('input_quota','Saved job input exceeds its configured limit.')
        job=_Job(**record,manifest=manifest,blobs=codec.blobs)
        _compatible(job.layout,job.packet['initial'])
        if job.pause_state is not None: _compatible(job.layout,job.pause_state['physical'])
        if job.status not in _TERMINAL:
            job.status='interrupted'; job.result=None
            job.error={'code':'coordinator_lost','message':'Previous coordinator ended; explicit physical restart is required.'}
        records[identity]=job
    with manager._condition:
        manager.config=config; manager.records=records; manager.runtime_id=registry['runtime_id']
        manager._admit([])

def flush(terminal):
    """Publish system transitions through the same single-writer recorder."""
    service=terminal.service; store=service._persistence.store
    if store.index.get('jobs',{}).get('signature')==signature(service._jobs): return
    from optimizer.persistence.controller import _clone
    from optimizer.operations.contracts import OperationResponse, OperationRequest
    import copy
    def publish(staged):
        p=staged._persistence; p.staged=_clone(staged.session); p.index=copy.deepcopy(p.store.index)
        p.files={}; p.track=False; p.guard=lambda:None
        before=staged.session.snapshot()
        p._sync(p.staged)
        response=OperationResponse('coordinator-transition','jobs.transition','ok',state_before=before,state_after=before,
            metadata={'summary':{'jobs':[{ 'job_id':key,'status':job.status} for key,job in service._jobs.records.items()]}})
        terminal.history.capture(response,OperationRequest('jobs.transition'))
        terminal.history.events[-1]['origin']='coordinator'
        return 0
    terminal.history.run('',None,publish,origin='coordinator')

def _read_output(directory, filename, limit):
    path=directory/filename
    total=path.stat().st_size
    if total>limit: raise ValueError('Oversized worker boundary.')
    blobs={}
    def read(digest):
        nonlocal total
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest): raise ValueError('Invalid blob identity.')
        if digest not in blobs:
            p=directory/'output_blobs'/digest; total+=p.stat().st_size
            if total>limit: raise ValueError('Worker boundary exceeds result reservation.')
            blobs[digest]=p.read_bytes()
        return blobs[digest]
    value=_Codec(read,max_bytes=limit).decode(json.loads(path.read_bytes()))
    if _size(value)>2*limit: raise ValueError('Decoded worker boundary exceeds reservation.')
    return value
