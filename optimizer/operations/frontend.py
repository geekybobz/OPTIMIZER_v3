"""Typed frontend transactions. Neither Python source nor CLI statements execute here."""
from __future__ import annotations
import inspect
import dataclasses
import math
from uuid import uuid4
import numpy as np
from optimizer.operations.contracts import ObjectRef, OperationRequest, OperationResponse, _ServiceError
from optimizer.checkpoints.codec import _json
from optimizer.cli.display import _describe
from optimizer.serialization import to_plain

INPUT_BYTES=65536
OUTPUT_BYTES=65536

def _bounded(value):
    nodes=0
    def walk(item,depth=0):
        nonlocal nodes
        nodes+=1
        if depth>24 or nodes>8192: raise _ServiceError('input_limit','Typed input exceeds nesting/node budget.')
        if item is None or type(item) in (str,bool,int): return
        if type(item) is float and math.isfinite(item): return
        if isinstance(item,list):
            for child in item: walk(child,depth+1)
            return
        if isinstance(item,dict) and all(isinstance(key,str) for key in item):
            for child in item.values(): walk(child,depth+1)
            return
        raise _ServiceError('invalid_request','Use finite JSON values and explicit object references.')
    walk(value)
    if len(_json(value))>INPUT_BYTES: raise _ServiceError('input_limit','Typed request exceeds 64 KiB.')

def _decode(value):
    if isinstance(value,list): return [_decode(child) for child in value]
    if not isinstance(value,dict): return value
    if '$ref' in value:
        if set(value)!={'$ref'} or not isinstance(value['$ref'],dict) or set(value['$ref'])-{'object_id','path'}:
            raise _ServiceError('invalid_reference','Use {"$ref":{"object_id":"name-or-id","path":[]}}.')
        ref=value['$ref']
        path=ref.get('path',[])
        if not isinstance(path,list): raise _ServiceError('invalid_reference','Reference path must be an array.')
        return ObjectRef(ref.get('object_id'),tuple(path))
    if '$tuple' in value:
        if set(value)!={'$tuple'} or not isinstance(value['$tuple'],list): raise _ServiceError('invalid_request','Invalid tuple envelope.')
        return tuple(_decode(child) for child in value['$tuple'])
    if any(key.startswith('$') for key in value): raise _ServiceError('invalid_request','Unknown typed envelope; imports, callbacks and pickle are unavailable.')
    return {key:_decode(child) for key,child in value.items()}

def _request(payload):
    if not isinstance(payload,dict) or set(payload)-{'operation','arguments','context','bind','activate','request_id','expected_revision','experiment_id'}:
        raise _ServiceError('invalid_request','Unknown operation request fields.')
    for name in ('operation','request_id','context','bind','experiment_id'):
        value=payload.get(name)
        if isinstance(value,str) and len(value.encode('utf-8'))>256:
            raise _ServiceError('input_limit',f'{name} exceeds 256 UTF-8 bytes.')
    return OperationRequest(**{key:_decode(value) for key,value in payload.items() if key not in {'expected_revision','experiment_id'}})

def capacity_request(action,payload,store):
    if action!='execute': return None
    try:
        _bounded(payload); request=_request(payload)
        from optimizer.history.recorder import _Recorder
        return request if _Recorder._capacity_override('',store,request) is not None else None
    except (TypeError,ValueError,_ServiceError): return None

def _descriptor(service,name):
    result=service.info(name)
    supported=result['classification'] in {'tool','constructor','lifecycle'} and name!='session.attach'
    result={**result,'transport':{'supported':supported,'arguments':'finite JSON, $ref and $tuple',
        'limitation':'Live callbacks/systems/renderers require native Python.',
        'reason':result.get('reason') if not supported else None}}
    if name=='session.attach': result['transport']['reason']='A live system cannot cross the transport; use session.load with an explicit source path.'
    target=service._session_methods().get(name) or service._persistence.methods.get(name)
    if service._history: target=target or service._history.methods.get(name)
    if target is None:
        item,_=service._registry.resolve(name); target=item.public_object()
    try:
        result['parameters']=[{'name':p.name,'required':p.default is inspect.Parameter.empty,
            'kind':p.kind.name,'default':None if p.default is inspect.Parameter.empty else _describe(p.default)}
            for p in inspect.signature(target).parameters.values()]
    except (TypeError,ValueError): result['parameters']=[]
    return result

def _inspection(value,detail,offset,limit):
    if detail=='summary': return {'summary':to_plain(_describe(value))}
    if detail!='values' or type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=1024:
        raise _ServiceError('invalid_arguments','Inspection requires detail summary/values, offset >= 0 and limit 1..1024.')
    from optimizer.controls import Controls
    if isinstance(value,Controls): value=value.as_matrix(copy=False)
    if isinstance(value,np.ndarray):
        items=value.reshape(-1)[offset:offset+limit].tolist(); size=value.size
        return {'type':'ndarray','shape':list(value.shape),'dtype':str(value.dtype),'size':size,
            'offset':offset,'items':to_plain(items),'next_offset':offset+len(items) if offset+len(items)<size else None}
    if isinstance(value,(list,tuple,dict)):
        sequence=list(value.items()) if isinstance(value,dict) else value
        items=sequence[offset:offset+limit]
        values={str(key):_describe(item) for key,item in items} if isinstance(value,dict) else [_describe(item) for item in items]
        return {'type':type(value).__name__,'size':len(sequence),'offset':offset,'items':to_plain(values),
                'next_offset':offset+len(items) if offset+len(items)<len(sequence) else None}
    return {'value':to_plain(_describe(value))}

def dispatch(history,action,payload,*,origin='mcp'):
    """Publish exactly one typed frontend request through the existing recorder."""
    root=history.service; before=root.session.snapshot(); response=None; request=None
    request_id=uuid4().hex
    bounded=False
    try:
        _bounded(payload);bounded=True
        if not isinstance(payload,dict): raise _ServiceError('invalid_request','Tool arguments must be an object.')
        if action=='execute':
            request=_request(payload); request_id=request.request_id
        elif action not in {'list','info','inspect','attach','detach'}:
            raise _ServiceError('unknown_operation','Unknown typed frontend action.')
    except (ValueError,TypeError,_ServiceError) as exc:
        # Bounded invalid input is still an experiment history outcome.
        invalid={'code':getattr(exc,'code','invalid_request'),'message':str(exc)}
        if not bounded: payload={'rejected':invalid}
        action='invalid'
    else: invalid=None

    def apply(service):
        nonlocal response
        start=service.session.snapshot()
        try:
            if invalid: raise _ServiceError(invalid['code'],invalid['message'])
            store=service._persistence.store
            if payload.get('experiment_id') is not None and (store is None or payload['experiment_id']!=store.project['id']):
                raise _ServiceError('experiment_changed','Request belongs to another experiment.')
            expected=payload.get('expected_revision')
            if expected is not None:
                if type(expected) is not int or expected<0: raise _ServiceError('invalid_request','expected_revision must be a nonnegative integer.')
                service.session._check_revision(expected)
            if action=='execute':
                if request.operation=='session.attach': raise _ServiceError('adapter_unavailable','Load an explicit source-backed system.')
                response=service.execute(request)
            else:
                allowed={'list':{'group','classification','offset','limit'},'info':{'operation'},
                         'inspect':{'reference','detail','offset','limit'},'attach':set(),'detach':set()}[action]
                if set(payload)-allowed: raise _ServiceError('invalid_arguments','Unknown frontend arguments.')
                if action=='list':
                    offset=payload.get('offset',0); limit=payload.get('limit',25)
                    if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=50:
                        raise _ServiceError('invalid_arguments','List offset >= 0 and limit 1..50.')
                    entries=[entry for entry in service.list() if (payload.get('group') is None or entry['id'].split('.')[0]==payload['group']) and
                             (payload.get('classification') is None or entry['classification']==payload['classification'])]
                    value={'operations':[dict(id=e['id'],classification=e['classification'],effect=e.get('effect'),reason=e.get('reason')) for e in entries[offset:offset+limit]],
                           'total':len(entries),'next_offset':offset+limit if offset+limit<len(entries) else None}
                elif action=='info': value=_descriptor(service,payload['operation'])
                elif action=='inspect':
                    reference=_decode({'$ref':payload['reference']})
                    value=_inspection(service.session.get(reference),payload.get('detail','summary'),payload.get('offset',0),payload.get('limit',128))
                else: value={'attached':action=='attach','project_id':store.project['id'] if store else None,'lifetime':'coordinator'}
                response=OperationResponse(request_id,'mcp.'+action,'ok',value,state_before=start,state_after=start,metadata={'summary':value})
            history.capture(response,request)
            if response.ok:
                summary=to_plain(_describe(response.result)) if action=='execute' else response.result
                response=dataclasses.replace(response,metadata={**response.metadata,'summary':summary})
            return 0 if response.ok else 2
        except (ValueError,TypeError,KeyError,_ServiceError) as exc:
            response=OperationResponse(request_id,request.operation if request else 'mcp.'+action,'error',state_before=start,state_after=start,
                error={'code':getattr(exc,'code','invalid_arguments'),'message':str(exc)})
            history.capture(response,request);return 2
    expression={'action':action,'request':payload}
    try:
        history.run('',expression,apply,origin=origin,request=request)
    except (OSError,ValueError,TypeError,_ServiceError) as exc:
        response=OperationResponse(request_id,request.operation if request else 'mcp.'+action,'error',state_before=before,state_after=root.session.snapshot(),
            error={'code':getattr(exc,'code','history_failed'),'message':str(exc)})
    store=root._persistence.store
    row=response.to_dict()
    row['metadata'].update(command_id=history.command_id,experiment_id=store.project['id'] if store else None,
                           durability='persistent' if store else 'ephemeral',durability_warning=store.warning if store else None)
    if len(_json(row))>OUTPUT_BYTES:
        row['metadata']={'omitted':'Output exceeds 64 KiB; inspect narrower fields or smaller pages.','command_id':history.command_id,
                         'experiment_id':store.project['id'] if store else None}
        row['state_before']=history._state(response.state_before);row['state_after']=history._state(response.state_after)
        if row['error']: row['error']['message']=row['error'].get('message','')[:2048]
        if len(_json(row))>OUTPUT_BYTES:
            row['state_before']={'revision':response.state_before.get('revision')}
            row['state_after']={'revision':response.state_after.get('revision')}
    return row
