"""Versioned, allowlisted data codec. No artifact-controlled imports or pickle."""
from __future__ import annotations
import dataclasses
import hashlib
import io
import json
import math
import zipfile
import numpy as np
from optimizer.controls import Controls, ControlSpec
from optimizer.core.record import RunRecord
from optimizer.utils.control_tools import ControlGrid, ControlTransformResult
from optimizer.jobs.contracts import JobHandle, JobsConfig
from optimizer.result import Evaluation, OptimizerResult
from optimizer.state import RunState, WarmStartState
from optimizer.system_olss.variables import ShootingVariables, ShootingSpec, ShootingBlock
from optimizer.system_olss import results as shooting_results
from optimizer.schedules import step_size as schedules
from optimizer.operations.contracts import ObjectRef, _ServiceError
from optimizer.visualization.models import PlotData, PlotLayer

_TYPES = {c.__name__: c for c in (RunRecord, ControlSpec, ControlGrid, ControlTransformResult, JobHandle, JobsConfig, Evaluation, OptimizerResult, RunState, WarmStartState,
                                ShootingVariables, ShootingSpec, ShootingBlock, ObjectRef, PlotData, PlotLayer)}
_TYPES.update({name: c for name, c in vars(shooting_results).items()
               if isinstance(c, type) and dataclasses.is_dataclass(c) and c.__module__ == shooting_results.__name__})
_TYPES.update({name: c for name, c in vars(schedules).items()
               if isinstance(c, type) and dataclasses.is_dataclass(c) and c.__module__ == schedules.__name__})

def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

def _hash(payload):
    return hashlib.sha256(payload).hexdigest()

class _Codec:
    def __init__(self, read_blob=None, *, max_bytes=64*1024*1024):
        self.blobs = {}
        self.read_blob = read_blob
        self.max_bytes = max_bytes
        self.decoded_bytes = 0
        self.nodes = 0

    def encode(self, value, depth=0):
        self.nodes += 1
        if depth > 64 or self.nodes > 100000:
            raise _ServiceError('codec_limit', 'Checkpoint nesting/node limit exceeded.')
        if isinstance(value, np.generic):
            return self.encode(value.item(), depth+1)
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            return value if math.isfinite(value) else {'$float': str(value)}
        if type(value) is complex:
            return {'$complex': [self.encode(value.real), self.encode(value.imag)]}
        if isinstance(value, bytes):
            return {'$bytes': value.hex()}
        if isinstance(value, np.dtype):
            return {'$dtype': value.str}
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in 'biufc' or value.nbytes > self.max_bytes:
                raise _ServiceError('unsupported_codec', 'Array must be bounded numeric non-object data.')
            array = np.array(value, copy=True, order='C', subok=False)
            header = {'dtype': array.dtype.str, 'shape': list(array.shape)}
            digest = _hash(_json(header) + array.tobytes())
            if digest not in self.blobs:
                buffer = io.BytesIO()
                np.savez(buffer, value=array)
                self.blobs[digest] = buffer.getvalue()
            return {'$array': digest, **header}
        if isinstance(value, Controls):
            return {'$controls': self.encode({'spec': value.spec, 'matrix': value.as_matrix(),
                                              'name': value.name, 'meta': value.meta}, depth+1)}
        if isinstance(value, type) and value.__name__ in _TYPES and _TYPES[value.__name__] is value:
            return {'$class': value.__name__}
        if dataclasses.is_dataclass(value) and type(value).__name__ in _TYPES and _TYPES[type(value).__name__] is type(value):
            fields = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
            return {'$type': type(value).__name__, 'fields': self.encode(fields, depth+1)}
        if isinstance(value, dict):
            return {'$dict': [[self.encode(k, depth+1), self.encode(v, depth+1)] for k,v in value.items()]}
        if isinstance(value, (tuple, list)):
            return {'$tuple' if isinstance(value, tuple) else '$list': [self.encode(v, depth+1) for v in value]}
        raise _ServiceError('unsupported_codec', f'No checkpoint codec for {type(value).__name__}.')

    def decode(self, value, depth=0):
        self.nodes += 1
        if depth > 64 or self.nodes > 100000:
            raise _ServiceError('codec_limit', 'Decoded nesting/node limit exceeded.')
        if value is None or type(value) in (str, bool, int, float):
            return value
        if not isinstance(value, dict):
            raise _ServiceError('checkpoint_corrupt', 'Expected tagged data.')
        keys = set(value)
        if keys == {'$float'} and value['$float'] in ('inf','-inf','nan'):
            return float(value['$float'])
        if keys == {'$complex'}:
            return complex(*[self.decode(v, depth+1) for v in value['$complex']])
        if keys == {'$bytes'}:
            return bytes.fromhex(value['$bytes'])
        if keys == {'$dtype'} and np.dtype(value['$dtype']).kind in 'biufc':
            return np.dtype(value['$dtype'])
        if keys == {'$class'} and value['$class'] in _TYPES:
            return _TYPES[value['$class']]
        if keys == {'$dict'}:
            return {self.decode(k, depth+1): self.decode(v, depth+1) for k,v in value['$dict']}
        if keys in ({'$tuple'}, {'$list'}):
            result = [self.decode(v, depth+1) for v in next(iter(value.values()))]
            return tuple(result) if '$tuple' in value else result
        if keys == {'$controls'}:
            fields = self.decode(value['$controls'], depth+1)
            return Controls.from_matrix(fields['spec'], fields['matrix'], name=fields['name'], meta=fields['meta'])
        if keys == {'$type', 'fields'} and value['$type'] in _TYPES:
            cls = _TYPES[value['$type']]
            fields = self.decode(value['fields'], depth+1)
            # One-way data migration only: never import or rebuild retired loggers.
            retired = {'trace_id', 'checkpoint_ids'}
            if cls is OptimizerResult: retired |= {'trace', 'tracking_failures'}
            if cls in (OptimizerResult, RunState, WarmStartState) and retired <= set(fields):
                for name in retired: fields.pop(name)
                if cls is OptimizerResult: fields['diagnostics'] = {}
            if cls in (OptimizerResult, shooting_results.ShootingResult) and "run_record" not in fields:
                fields["run_record"] = None
            if cls is shooting_results.ShootingResult and "state" not in fields:
                fields["state"] = None
            if cls is ControlSpec and "scales" not in fields:
                fields["scales"] = None
            if set(fields) != {f.name for f in dataclasses.fields(cls)}:
                raise _ServiceError('checkpoint_corrupt', 'Dataclass fields do not match codec version.')
            return cls(**fields)
        if keys == {'$array','dtype','shape'}:
            dtype = np.dtype(value['dtype'])
            shape = value['shape']
            if dtype.kind not in 'biufc' or not isinstance(shape,list) or any(type(n) is not int or n < 0 for n in shape):
                raise _ServiceError('checkpoint_corrupt', 'Invalid array layout.')
            size = math.prod(shape)*dtype.itemsize
            self.decoded_bytes += size
            if len(shape)>32 or self.decoded_bytes > self.max_bytes:
                raise _ServiceError('codec_limit', 'Decoded arrays exceed RAM budget.')
            payload = self.read_blob(value['$array'])
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                entries = archive.infolist()
                if len(entries)!=1 or entries[0].filename!='value.npy' or entries[0].file_size > size+4096:
                    raise _ServiceError('checkpoint_corrupt', 'Invalid/oversized array container.')
                # Validate the actual NPY header before numpy allocates from it.
                with archive.open(entries[0]) as handle:
                    version=np.lib.format.read_magic(handle)
                    reader={(1,0):np.lib.format.read_array_header_1_0,
                            (2,0):np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None: raise _ServiceError('checkpoint_corrupt','Unsupported NPY header version.')
                    actual, _, actual_dtype=reader(handle,max_header_size=4096)
                if actual != tuple(shape) or actual_dtype != dtype:
                    raise _ServiceError('checkpoint_corrupt','Array header disagrees with manifest.')
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                array = archive['value']
            if _hash(_json({'dtype':dtype.str,'shape':shape}) + np.ascontiguousarray(array).tobytes()) != value['$array']:
                raise _ServiceError('checkpoint_corrupt', 'Array hash mismatch.')
            return array
        raise _ServiceError('checkpoint_corrupt', 'Unknown mandatory codec tag/type.')
