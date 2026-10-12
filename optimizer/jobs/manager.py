"""One bounded runtime coordinator; never writes foreground history from a worker."""
from __future__ import annotations

import atexit
import dataclasses
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from uuid import uuid4

from optimizer.jobs.contracts import JobHandle, JobsConfig
from optimizer.operations.contracts import ObjectRef, _ServiceError
from optimizer.serialization import to_plain
from optimizer.session import _compatible, _owned, _physical, _size, _Context

_TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def _refs(value):
    if isinstance(value, ObjectRef): return (value,)
    if isinstance(value, dict): return tuple(r for v in value.values() for r in _refs(v))
    if isinstance(value, (tuple, list)): return tuple(r for v in value for r in _refs(v))
    return ()


def _rss(process):
    """Sample only an owned live child; unsupported platforms report unavailable."""
    if process.poll() is not None or os.name != "posix": return None
    try:
        raw = subprocess.check_output(["ps", "-o", "rss=", "-p", str(process.pid)],
                                      stderr=subprocess.DEVNULL, timeout=1)
        return int(raw.strip()) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


@dataclasses.dataclass
class _Job:
    handle: JobHandle
    packet: dict
    manifest: bytes
    blobs: dict
    sources: dict
    source_path: str
    recipe: dict
    layout: object
    family: str
    input_refs: tuple
    worker_bytes: int
    result_bytes: int
    cache_bytes: int
    scratch_bytes: int
    status: str = "queued"
    process: object = None
    directory: object = None
    result: object = None
    error: object = None
    cancel_requested: bool = False
    cancel_time: float | None = None
    peak_rss: int | None = None
    forced: bool = False
    cancellation_boundary: str | None = None


class _Jobs:
    def __init__(self):
        self.runtime_id = uuid4().hex
        self.config = JobsConfig()
        self.records = {}
        self._condition = threading.Condition(threading.RLock())
        self._thread = None
        self._closed = False
        # Bound service lifetime, including ephemeral CLI EOF. No durable daemon.
        self._registered_exit = False

    def _get(self, handle):
        if isinstance(handle, JobHandle):
            if handle.runtime_id != self.runtime_id:
                raise _ServiceError("job_interrupted", "This handle belongs to a closed/lost runtime; no numerical restart occurs.")
            identity = handle.job_id
        elif isinstance(handle, str): identity = handle
        else: raise _ServiceError("invalid_arguments", "job must be JobHandle or job ID.")
        job = self.records.get(identity)
        if job is None: raise _ServiceError("unknown_job", "Job is missing or explicitly released.")
        if isinstance(handle, JobHandle) and job.handle != handle:
            raise _ServiceError("invalid_arguments", "Job handle differs from its submitted identity.")
        return job

    def _status(self, job):
        return {"job":job.handle, "status":job.status, "cancel_requested":job.cancel_requested,
                "error":_owned(job.error), "result_available":job.result is not None,
                "worker_bytes":job.worker_bytes, "result_bytes":job.result_bytes,
                "peak_rss":job.peak_rss, "forced":job.forced,
                "cancellation_boundary":job.cancellation_boundary,
                "process_id":job.process.pid if job.process is not None else None,
                "lifetime":"client", "resume_level":"physical_only"}

    def _admit(self, pending):
        cfg = self.config
        jobs = [*self.records.values(), *pending]
        if len(jobs) > cfg.max_jobs:
            raise _ServiceError("job_quota", "Release completed jobs before submitting more.")
        active = [j for j in jobs if j.status not in _TERMINAL]
        # Admission simulates the slots that fit RAM as well as CPU; a large
        # worker estimate must not silently create an oversized effective queue.
        occupied=[j for j in active if j.status=="running"]
        available=cfg.memory_bytes-cfg.coordinator_bytes-sum(j.worker_bytes for j in occupied)
        slots=len(occupied)
        for j in active:
            if j.status=="queued" and slots<cfg.max_workers and j.worker_bytes<=available:
                available-=j.worker_bytes; slots+=1
        if len(active)-slots > cfg.max_queue:
            raise _ServiceError("queue_full", "Worker slots and bounded queue are full.")
        if sum(j.cache_bytes for j in jobs) > cfg.coordinator_bytes:
            raise _ServiceError("memory_admission", "Frozen input/result reservations exceed coordinator_bytes.")
        if sum(j.scratch_bytes for j in active) > cfg.scratch_bytes:
            raise _ServiceError("scratch_admission", "Prepared inputs/results exceed scratch_bytes.")
        if any(j.worker_bytes > cfg.memory_bytes - cfg.coordinator_bytes for j in active):
            raise _ServiceError("memory_admission", "Worker estimate cannot fit alongside the coordinator reserve.")
        if sum(j.scratch_bytes for j in active) > shutil.disk_usage(tempfile.gettempdir()).free:
            raise _ServiceError("scratch_admission", "Insufficient temporary disk capacity.")

    def _prepare(self, service, operation, controls, variables, options, label, worker_bytes, result_bytes, context):
        from optimizer.checkpoints.codec import _Codec, _hash, _json
        from optimizer.core.engine import run_chunk
        from optimizer.persistence.controller import _Persistence
        c = service.session._context(context)
        if not c.recipe.get("path") or c.recipe.get("checkpoint", {}).get("stateless") is not True:
            raise _ServiceError("unsupported_factory", "Jobs need session.load(source_path, checkpoint={stateless:True, dependencies:[...]}).")
        if "dependencies" not in c.recipe.get("checkpoint", {}):
            raise _ServiceError("unsupported_factory", "Declare the factory's local dependencies explicitly.")
        for name, value in (("worker_bytes",worker_bytes),("result_bytes",result_bytes)):
            if type(value) is not int or value < 4096:
                raise _ServiceError("invalid_arguments", f"{name} must be an integer >= 4096.")
        if label is not None and (not isinstance(label,str) or not label or len(label)>128):
            raise _ServiceError("invalid_arguments", "label must be a nonempty string of at most 128 characters.")
        item, adapter = service._registry.resolve(operation)
        if not (item.group == "optimizers" or item.id in {"shooting.solve_shooting", "shooting.solve_direct_shooting"}):
            raise _ServiceError("unsupported_job", "Submit a registered OLGS optimizer or OLSS shooting solver.")
        if c.family not in adapter.family:
            raise _ServiceError("wrong_family", "Solver and submitted context have different families.")
        if controls is not None and variables is not None:
            raise _ServiceError("invalid_arguments", "Supply controls or variables, not both.")
        if (c.family=="olgs" and variables is not None) or (c.family=="olss" and controls is not None):
            raise _ServiceError("wrong_family", "Use controls for OLGS and variables for OLSS.")
        initial = controls if controls is not None else variables
        if initial is None:
            initial = _physical(service.session.get(c.head)) if c.head else None
        if initial is None: raise _ServiceError("missing_state", "Initialize controls/variables before submitting.")
        _compatible(c.layout, initial)
        initial = _owned(_physical(initial))
        options = {} if options is None else dict(options)
        signature = inspect.signature(item.public_object())
        allowed = set(signature.parameters)
        if "engine_kwargs" in allowed: allowed |= set(inspect.signature(run_chunk).parameters)
        forbidden = {"system","controls","initial","variables","state","warmstart","previous_result","evaluator",
                     "system_factory","parallel","on_event","on_iteration","step","engine_kwargs"}
        unknown = set(options)-allowed
        if unknown or set(options)&forbidden:
            raise _ServiceError("invalid_arguments", f"Invalid job options: {sorted(unknown | (set(options)&forbidden))}.")
        if type(options.get("maxiter")) is not int or options["maxiter"] < 0:
            raise _ServiceError("invalid_arguments", "Declare maxiter as a nonnegative integer.")
        if item.name == "cma_es" and type(options.get("seed")) is not int:
            raise _ServiceError("invalid_arguments", "Stochastic CMA-ES jobs require an explicit integer seed.")
        recipe = _owned(c.recipe)
        paths=[Path(recipe["path"])]
        for dependency in recipe["checkpoint"]["dependencies"]:
            path=Path(dependency).expanduser()
            paths.append(path if path.is_absolute() else paths[0].parent/path)
        if sum(path.stat().st_size for path in set(paths)) > self.config.input_bytes:
            raise _ServiceError("input_quota", "Source bundle exceeds input_bytes before reading.")
        frozen = _Persistence(service)._recipe_files(recipe)
        source_paths = [Path(recipe["path"]), *(Path(p) for p in recipe.get("dependency_hashes", {}))]
        common = Path(os.path.commonpath([str(p.parent) for p in source_paths]))
        sources = {}
        for path in source_paths:
            digest = recipe["sha256"] if str(path)==recipe["path"] else recipe["dependency_hashes"][str(path)]
            sources[str(path.relative_to(common))] = frozen[f"inputs/{digest}/payload"]
        source_path = str(Path(recipe["path"]).relative_to(common))
        system_spec = to_plain(c.system.system_spec())
        fingerprint = _hash(_json({"system_spec":system_spec,"sources":{n:_hash(b) for n,b in sources.items()}}))
        handle = JobHandle(uuid4().hex, self.runtime_id, item.id, c.name, c.configuration, fingerprint, label)
        packet = {"recipe":recipe,"family":c.family,"initial":initial,"options":options,
                  "operation":item.id,"source_path":source_path,"system_spec":system_spec}
        codec = _Codec(max_bytes=self.config.input_bytes)
        manifest = _json(codec.encode(packet))
        input_size = len(manifest) + sum(map(len,codec.blobs.values())) + sum(map(len,sources.values()))
        if input_size > self.config.input_bytes:
            raise _ServiceError("input_quota", "Frozen input/source bundle exceeds input_bytes.")
        # Verify the payload now, before admitting or starting a worker.
        _Codec(codec.blobs.__getitem__,max_bytes=self.config.input_bytes).decode(json.loads(manifest))
        pending = [action[1] for action in service._job_actions if action[0]=="submit"]
        refs = _refs(service._job_original_arguments)
        if controls is None and variables is None and c.head: refs += (ObjectRef(c.head),)
        job = _Job(handle, packet, manifest, codec.blobs, sources, source_path, recipe, _owned(c.layout), c.family,
                   refs, worker_bytes, result_bytes,
                   _size(packet)+_size(manifest)+_size(codec.blobs)+_size(sources)+4*result_bytes+8192,
                   input_size+2*result_bytes)
        with self._condition: self._admit([*pending, job])
        return job

    def dispatch(self, request, service):
        from optimizer.jobs import api
        from optimizer.operations.service import _summary
        operation = request.operation
        name = operation.split(".")[-1]
        obj = getattr(api, name, None)
        if obj is None or name.startswith("_"):
            raise _ServiceError("unknown_operation", operation)
        if request.activate is not None or "service" in request.arguments:
            raise _ServiceError("invalid_arguments", "Jobs never activate state or accept a live service argument.")
        before = service.session.snapshot()
        arguments = service._resolve(dict(request.arguments))
        if request.context is not None:
            if name != "submit" or "context" in arguments:
                raise _ServiceError("invalid_arguments", "context is accepted once for jobs.submit only.")
            arguments["context"] = request.context
        if name != "configure":
            try:
                bound = inspect.signature(obj).bind(**arguments)
                bound.apply_defaults()
                arguments = {k:v for k,v in bound.arguments.items() if k != "service"}
            except TypeError as exc: raise _ServiceError("invalid_arguments",str(exc)) from exc
        else:
            unknown=set(arguments)-{f.name for f in dataclasses.fields(JobsConfig)}
            if unknown: raise _ServiceError("invalid_arguments",f"Unknown limits: {sorted(unknown)}.")
        with self._condition:
            source = None
            inputs = _refs(request.arguments)
            if name == "configure":
                if self.records or service._job_actions:
                    raise _ServiceError("jobs_active", "Release all jobs before changing scheduler limits.")
                configured = dataclasses.replace(self.config, **arguments)
                service._job_actions.append(("configure", configured))
                value = dataclasses.asdict(configured)
            elif name == "submit":
                service._job_original_arguments = request.arguments
                source = self._prepare(service, **arguments)
                service._job_actions.append(("submit",source))
                inputs = source.input_refs
                value = source.handle
            elif name == "status" and arguments["job"] is None:
                value = {"runtime_id":self.runtime_id, "config":dataclasses.asdict(self.config),
                         "jobs":[self._status(j) for j in self.records.values()],
                         "running":sum(j.status=="running" for j in self.records.values()),
                         "queued":sum(j.status=="queued" for j in self.records.values()),
                         "cache_reserved":sum(j.cache_bytes for j in self.records.values()),
                         "memory_enforcement":"sampled RSS cancellation on POSIX; reservations elsewhere",
                         "native_thread_enforcement":"environment hints, project/native libraries must honor them",
                         "backend":"process", "lifetime":"client"}
            elif name == "shutdown":
                service._job_actions.append(("shutdown",None))
                value = {"shutdown_requested":True,"jobs":len(self.records),"lifetime":"client"}
            elif name == "compare":
                value = self._compare(**arguments)
            else:
                source = self._get(arguments["job"])
                if name == "wait":
                    timeout=float(arguments["timeout"])
                    if not math.isfinite(timeout) or not 0 <= timeout <= 30:
                        raise _ServiceError("invalid_arguments","timeout must be finite and between 0 and 30 seconds.")
                    deadline=time.monotonic()+timeout
                    while source.status not in _TERMINAL and time.monotonic()<deadline:
                        self._condition.wait(min(.1, max(0,deadline-time.monotonic())))
                    value=self._status(source)
                elif name == "status": value=self._status(source)
                elif name == "cancel":
                    service._job_actions.append(("cancel",source.handle.job_id))
                    value={**self._status(source),"cancel_requested":source.status not in _TERMINAL or source.cancel_requested}
                elif name == "release":
                    if source.status not in _TERMINAL: raise _ServiceError("job_not_ready","Cancel/wait before releasing a job.")
                    service._job_actions.append(("release",source.handle.job_id))
                    value={"released":source.handle.job_id}
                elif name == "result":
                    if source.status not in _TERMINAL: raise _ServiceError("job_not_ready","Worker is queued/running; use status or wait.")
                    if source.result is None:
                        raise _ServiceError("job_failed",f"{source.status}: {source.error}")
                    value=_owned(source.result)
                else: raise _ServiceError("unknown_operation",operation)
        context = None
        if name == "result":
            # Original provenance only; method state is deliberately uncertified.
            context=_Context(source.handle.context,None,source.family,source.layout,None,source.recipe,
                             configuration=source.handle.configuration)
            inputs=source.input_refs
        ref=service.session._publish(value,revision=before["revision"],bind=request.bind,context=context,inputs=inputs)
        metadata={"summary":_summary(value),"activated":False,"execution_state_before":before,
                  "resume_level":"physical_only" if name=="result" else "analysis_only"}
        if name=="result":
            current=service.session._contexts.get(source.handle.context)
            metadata.update(job=source.handle.job_id,job_status=source.status,
                            stale_configuration=current is None or current.configuration!=source.handle.configuration or
                            {k:v for k,v in current.recipe.items() if k!="dependency_hashes"} !=
                            {k:v for k,v in source.recipe.items() if k!="dependency_hashes"})
        return _owned(value),(ref,),metadata,operation

    def _compare(self, jobs, metric, direction):
        if direction not in {"minimize","maximize"} or not isinstance(metric,str) or not metric:
            raise _ServiceError("invalid_arguments","Declare a metric and minimize/maximize direction.")
        items=[self._get(handle) for handle in jobs]
        if not items or len({j.handle.job_id for j in items})!=len(items):
            raise _ServiceError("invalid_arguments","Compare a nonempty list of unique jobs.")
        if any(j.handle.fingerprint!=items[0].handle.fingerprint for j in items):
            raise _ServiceError("incompatible_comparison","Jobs must share the frozen problem declaration/source fingerprint.")
        rows=[]
        for index,j in enumerate(items):
            metrics=getattr(j.result,"metrics",{})
            number=metrics.get(metric)
            try: number=float(number)
            except (TypeError,ValueError): number=None
            verified=getattr(j.result,"verified",None)
            qualified=getattr(j.result,"qualified",None)
            eligible=j.status=="succeeded" and number is not None and math.isfinite(number) and (
                j.family!="olss" or verified is True and qualified is True)
            rows.append({"job":j.handle,"status":j.status,"value":number if number is not None and math.isfinite(number) else None,
                         "eligible":eligible,"verified":verified,"qualified":qualified,"submission_order":index})
        ranked=sorted((r for r in rows if r["eligible"]),key=lambda r:(r["value"] if direction=="minimize" else -r["value"],r["submission_order"]))
        return {"metric":metric,"direction":direction,"rows":rows,"ranking":[r["job"] for r in ranked],"activated":False}

    def commit(self, actions):
        """Called after successful foreground/durable commit; never rollback work."""
        for kind,value in actions:
            if kind=="shutdown": self.close(); continue
            with self._condition:
                if kind=="configure": self.config=value; self._closed=False
                elif kind=="submit":
                    if self._closed: self._closed=False
                    self.records[value.handle.job_id]=value
                    try:
                        self._ensure_thread()
                    except Exception as exc:
                        value.status="failed"
                        value.error={"code":"worker_start_failed","message":str(exc)[:1024]}
                elif kind=="cancel":
                    job=self.records.get(value)
                    if job is not None: self._cancel(job)
                elif kind=="release":
                    job=self.records.pop(value,None)
                    if job is not None and job.directory: shutil.rmtree(job.directory,ignore_errors=True)
                self._condition.notify_all()

    def _ensure_thread(self):
        if not self._registered_exit:
            atexit.register(self.close)
            self._registered_exit = True
        if self._thread is None or not self._thread.is_alive():
            self._thread=threading.Thread(target=self._run,daemon=True,name="optimizer-jobs")
            self._thread.start()

    def _cancel(self, job):
        if job.status in _TERMINAL: return
        job.cancel_requested=True
        if job.status=="queued": job.status="cancelled"; return
        if job.cancel_time is None: job.cancel_time=time.monotonic()
        if job.directory:
            try: (job.directory/"cancel").touch()
            except OSError: pass

    def _start(self, job):
        try:
            directory=Path(tempfile.mkdtemp(prefix="optimizer-job-")); job.directory=directory
            (directory/"input.json").write_bytes(job.manifest)
            (directory/"input_blobs").mkdir()
            for digest,data in job.blobs.items(): (directory/"input_blobs"/digest).write_bytes(data)
            for relative,data in job.sources.items():
                path=directory/"source"/relative; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
            environment=dict(os.environ)
            environment.update({name:str(self.config.threads_per_worker) for name in _THREAD_ENV})
            # Preserve the currently running package, not a stale global install.
            package_root=Path(__file__).resolve().parents[2]
            environment["PYTHONPATH"]=str(package_root)
            environment["PYTHONDONTWRITEBYTECODE"]="1"
            job.process=subprocess.Popen([sys.executable,"-B","-m","optimizer.jobs.worker",str(directory),
                str(job.result_bytes),str(self.config.shutdown_seconds),str(self.config.input_bytes)],
                stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                cwd=directory,env=environment)
            job.status="running"
        except Exception as exc:
            job.status="failed"; job.error={"code":"worker_start_failed","message":str(exc)[:1024]}
            if job.directory: shutil.rmtree(job.directory,ignore_errors=True); job.directory=None

    def _finish(self, job):
        from optimizer.checkpoints.codec import _Codec
        try:
            path=job.directory/"output.json"
            if not path.exists() or path.stat().st_size>job.result_bytes:
                raise ValueError("Worker exited without a bounded complete outcome.")
            total=path.stat().st_size
            blobs={}
            def read(digest):
                nonlocal total
                if len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("Invalid worker blob identity.")
                if digest in blobs: return blobs[digest]
                blob=job.directory/"output_blobs"/digest
                total+=blob.stat().st_size
                if total>job.result_bytes: raise ValueError("Worker outcome exceeds result reservation.")
                blobs[digest]=blob.read_bytes()
                return blobs[digest]
            output=_Codec(read,max_bytes=job.result_bytes).decode(json.loads(path.read_bytes()))
            if output["status"] not in _TERMINAL: raise ValueError("Invalid worker terminal state.")
            if _size(output)>job.result_bytes*2: raise ValueError("Decoded result exceeds retained-data reservation.")
            physical=_physical(output.get("result"))
            if physical is not None: _compatible(job.layout,physical)
            job.status=output["status"]; job.result=output.get("result"); job.error=output.get("error") or job.error
            job.cancellation_boundary=output.get("cancellation_boundary")
            if job.error and job.error.get("code")=="worker_memory_limit" and job.status=="succeeded":
                job.status="failed"; job.result=None
            if job.forced:
                job.status="interrupted"; job.result=None
            if job.error is None and job.status=="interrupted":
                job.error={"code":"forced_interruption","message":"Worker did not acknowledge a safe boundary; no partial result retained."}
        except Exception as exc:
            job.status="interrupted"; job.result=None
            job.error=job.error or {"code":"worker_lost","message":str(exc)[:1024]}
        finally:
            if job.process.stdin: job.process.stdin.close()
            job.process.wait()
            shutil.rmtree(job.directory,ignore_errors=True); job.directory=None
            self._condition.notify_all()

    def _run(self):
        with self._condition:
            while not self._closed:
                running=[j for j in self.records.values() if j.status=="running"]
                for job in running:
                    if job.process.poll() is not None:
                        self._finish(job); continue
                    measured=_rss(job.process)
                    if measured is not None:
                        job.peak_rss=max(measured,job.peak_rss or 0)
                        if measured>job.worker_bytes:
                            job.error={"code":"worker_memory_limit","message":"Sampled RSS exceeded the declared worker estimate."}
                            self._cancel(job)
                    if job.cancel_time is not None and time.monotonic()-job.cancel_time>self.config.shutdown_seconds:
                        job.forced=True
                        try: job.process.kill()
                        except ProcessLookupError: pass
                running=[j for j in self.records.values() if j.status=="running"]
                memory=sum(j.worker_bytes for j in running)+self.config.coordinator_bytes
                for job in self.records.values():
                    if job.status=="queued" and len(running)<self.config.max_workers and memory+job.worker_bytes<=self.config.memory_bytes:
                        self._start(job)
                        if job.status=="running": running.append(job); memory+=job.worker_bytes
                if not any(j.status not in _TERMINAL for j in self.records.values()): return
                self._condition.wait(.1)

    def close(self):
        with self._condition:
            for job in self.records.values(): self._cancel(job)
            deadline=time.monotonic()+self.config.shutdown_seconds+.5
            while any(j.status=="running" for j in self.records.values()) and time.monotonic()<deadline:
                self._condition.wait(.1)
            for job in self.records.values():
                if job.status=="running":
                    job.forced=True
                    try: job.process.kill()
                    except ProcessLookupError: pass
                    self._finish(job)
            self._closed=True; self._condition.notify_all()
