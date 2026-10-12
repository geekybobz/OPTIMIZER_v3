"""Real coordinator processes, durable transitions and honest live continuation."""
from __future__ import annotations
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import optimizer as opt
from optimizer.cli.language import _Terminal
from optimizer.jobs.coordinator import start, request, _RemoteTerminal, _recv, _send, _identity
from optimizer.jobs.durable import flush
from optimizer.operations.contracts import _ServiceError
from optimizer.persistence.store import _Store
from optimizer.history.journal import _Journal
from fixtures.quadratic_system import QuadraticVectorSystem
from fixtures.shooting_systems import CubicRootShootingSystem

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tests/fixtures/job_system.py'
SHOT=ROOT/'tests/fixtures/job_shooting.py'

class PauseTests(unittest.TestCase):
    def setUp(self):
        self.service=opt.OperationService(); self.addCleanup(self.service.close)
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
    def call(self,name,args=None,**kwargs):
        response=self.service.execute(opt.OperationRequest(name,args or {},**kwargs))
        self.assertTrue(response.ok,response.error); return response
    def load(self):
        self.marker=Path(self.temp.name)/'calls.jsonl'
        self.call('session.load',{'name':'main'+str(len(self.service.session._contexts)),'factory':str(SOURCE),'parameters':{'N':5,'delay':.05,'marker':str(self.marker)},'checkpoint':{'stateless':True,'dependencies':['quadratic_system.py']}})
        return self.call('guesses.constant_guess',{'value':.1},activate=True).result
    def pause_after_work(self,job):
        deadline=time.monotonic()+6
        while time.monotonic()<deadline:
            if self.marker.exists() and len(self.marker.read_text().splitlines())>=3: break
            time.sleep(.02)
        else: self.fail('Worker did not perform numerical calls')
        opt.jobs.pause(job,service=self.service)
        self.assertEqual(opt.jobs.wait(job,timeout=10,service=self.service)['status'],'paused')
        snapshot=opt.jobs.checkpoint(job,service=self.service)
        self.assertGreaterEqual(snapshot['iteration'],1)
        return snapshot
    def test_adam_and_heavy_ball_live_resume_exact_numerical_parity(self):
        for method in ('adam','momentum'):
            with self.subTest(method=method):
                initial=self.load()
                options={'maxiter':12,'step_size':.02,'gradient_tolerance':None,'stall_patience':8,'stall_tolerance':0}
                job=opt.jobs.submit(method,options=options,service=self.service)
                snapshot=self.pause_after_work(job)
                self.assertEqual(snapshot['resume_level'],'physical_only')
                before=self.service.session.snapshot()
                resume=opt.jobs.resume(job,service=self.service)
                self.assertEqual(resume['resume_level'],'exact_live')
                self.assertEqual(self.service.session.snapshot()['contexts'],before['contexts'])
                self.assertEqual(opt.jobs.wait(job,timeout=10,service=self.service)['status'],'succeeded')
                result=opt.jobs.result(job,service=self.service)
                expected=getattr(opt,method)(QuadraticVectorSystem(N=5),initial,**options)
                np.testing.assert_array_equal(result.controls.u,expected.controls.u)
                self.assertEqual(result.metrics,expected.metrics)
                self.assertEqual(result.state.global_iteration,expected.state.global_iteration)
                self.assertEqual(result.stop_reason,expected.stop_reason)
                for key,value in result.state.optimizer_state.items():
                    if isinstance(value,np.ndarray): np.testing.assert_array_equal(value,expected.state.optimizer_state[key])
                opt.jobs.release(job,service=self.service)
                self.marker.unlink(missing_ok=True)
    def test_lm_live_pause_matches_native_variables_history_and_verification(self):
        self.call('session.load',{'name':'shot','factory':str(SHOT),'parameters':{'delay':.06},'checkpoint':{'stateless':True,'dependencies':['shooting_systems.py']}})
        initial=opt.ShootingVariables(self.service.session._context('shot').layout,[.6])
        self.call('session.use',{'value':initial})
        options={'maxiter':20,'tolerance':1e-12}
        job=opt.jobs.submit('shooting.solve_shooting',options=options,service=self.service)
        deadline=time.monotonic()+5
        while self.service._jobs.records[job.job_id].status=='queued' and time.monotonic()<deadline: time.sleep(.01)
        time.sleep(.25); opt.jobs.pause(job,service=self.service)
        status=opt.jobs.wait(job,timeout=10,service=self.service)
        self.assertEqual(status['status'],'paused',status)
        opt.jobs.resume(job,service=self.service)
        self.assertEqual(opt.jobs.wait(job,timeout=10,service=self.service)['status'],'succeeded')
        result=opt.jobs.result(job,service=self.service)
        expected=opt.solve_shooting(CubicRootShootingSystem(),initial,**options)
        np.testing.assert_array_equal(result.variables.values,expected.variables.values)
        self.assertEqual(result.history,expected.history)
        self.assertEqual(result.verified,expected.verified); self.assertEqual(result.qualified,expected.qualified)
    def test_physical_restart_has_new_identity_and_explicit_budget(self):
        self.load(); job=opt.jobs.submit('adam',options={'maxiter':20},service=self.service)
        snapshot=self.pause_after_work(job)
        with self.assertRaises(_ServiceError): opt.jobs.resume(job,mode='physical',service=self.service)
        child=opt.jobs.resume(job,mode='physical',maxiter=3,service=self.service)
        self.assertNotEqual(child.job_id,job.job_id)
        self.assertEqual(opt.jobs.wait(child,timeout=10,service=self.service)['status'],'succeeded')
        result=opt.jobs.result(child,service=self.service)
        expected=opt.adam(QuadraticVectorSystem(N=5),snapshot['physical'],maxiter=3)
        np.testing.assert_array_equal(result.controls.u,expected.controls.u)
        self.assertEqual(opt.jobs.status(child,service=self.service)['parent'],job)
    def test_unsupported_pause_and_invalid_resume_do_not_change_runtime(self):
        self.load(); job=opt.jobs.submit('cma_es',options={'maxiter':3,'seed':9},service=self.service)
        with self.assertRaises(_ServiceError) as error: opt.jobs.pause(job,service=self.service)
        self.assertEqual(error.exception.code,'unsupported_pause')
        with self.assertRaises(_ServiceError): opt.jobs.resume(job,mode='guess',service=self.service)
        with self.assertRaises(_ServiceError): opt.jobs.detach(service=self.service)
        self.assertFalse(self.service._jobs.records[job.job_id].cancel_requested)
    def test_pause_publication_failure_and_nested_error_have_no_runtime_effect(self):
        self.load(); terminal=_Terminal(self.service)
        job=opt.jobs.submit('adam',options={'maxiter':50},service=self.service)
        self.service.session.bind_reference('j',self.service.session._publish(job,revision=self.service.session.snapshot()['revision']))
        self.assertEqual(terminal.submit('opt.utils.scale_controls(opt.jobs.pause(j),factors=2);'),2)
        self.assertEqual(self.service._jobs.records[job.job_id].status,'running')

class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.path=self.root/'.optimizer/demo'
        terminal=_Terminal(); self.addCleanup(terminal.service.close)
        self.assertEqual(terminal.submit(f"opt.project.init('demo',root={str(self.root)!r});opt.session.load('main',{str(SOURCE)!r},parameters={{'N':5,'delay':.06}},checkpoint={{'stateless':True,'dependencies':['quadratic_system.py']}});u=opt.guesses.constant_guess(value=.1);opt.session.use(u);"),0)
        terminal.finish(); self.responses=[]
        start(self.path)
        self.client=_RemoteTerminal(self.path,emit=self.responses.append)
        self.addCleanup(self.stop)
    def stop(self):
        try: request(self.path,'submit',source='opt.jobs.shutdown();')
        except _ServiceError: pass
        deadline=time.monotonic()+5
        while (self.path/'runtime/coordinator.json').exists() and time.monotonic()<deadline: time.sleep(.02)
    def command(self,source,code=0):
        self.responses.clear()
        self.assertEqual(self.client.submit(source),code,self.responses[-1].error if self.responses else None)
        return self.responses[-1].metadata.get('summary') if self.responses else None
    def records(self):
        return [json.loads(line) for line in (self.path/'history.jsonl').read_text().splitlines()]
    def test_detach_reconnect_retains_running_jobs_and_commits_completion_once(self):
        self.command("j=opt.jobs.submit('adam',options={'maxiter':12});opt.jobs.detach();")
        self.assertTrue(self.client.closed)
        other=_RemoteTerminal(self.path,emit=self.responses.append); self.client=other
        self.command('opt.jobs.wait(j,timeout=10);r=opt.jobs.result(j);r.metrics;')
        rows=self.records()
        transitions=[event for row in rows if row['kind']=='outcome'
                     for event in row['data']['operations'] if event['operation']=='jobs.transition']
        self.assertTrue(transitions)
        self.assertTrue(all(event['origin']=='coordinator' for event in transitions))
        self.assertTrue(any(row['kind']=='submitted' and row['data']['origin']=='coordinator' and row['data']['raw']=='' for row in rows))
        count=sum(row['kind']=='submitted' and row['data']['origin']=='coordinator' for row in rows)
        time.sleep(.3)
        self.assertEqual(sum(row['kind']=='submitted' and row['data']['origin']=='coordinator' for row in self.records()),count)
        other.finish(); self.assertTrue((self.path/'runtime/coordinator.json').exists())
    def test_paused_checkpoint_survives_exit_and_resume_is_exact_live(self):
        self.command("j=opt.jobs.submit('adam',options={'maxiter':12});opt.jobs.pause(j);opt.jobs.wait(j,timeout=10);")
        status=self.command('opt.jobs.status(j);'); self.assertEqual(status['status'],'paused')
        self.client.finish(); self.client=_RemoteTerminal(self.path,emit=self.responses.append)
        self.command('opt.jobs.resume(j);opt.jobs.wait(j,timeout=10);r=opt.jobs.result(j);')
        self.assertEqual(self.command('r.metrics;')['J'],opt.adam(QuadraticVectorSystem(N=5),opt.constant_guess(QuadraticVectorSystem(N=5),value=.1),maxiter=12).metrics['J'])
    def test_single_writer_authentication_and_duplicate_request(self):
        with self.assertRaises(BlockingIOError): _Store(self.path)
        self.command('opt.project.close();',2)
        self.assertEqual(self.responses[-1].error['code'],'coordinator_owned')
        _,identity=_identity(self.path)
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
            sock.connect(identity['socket']); _send(sock,{'token':'bad','identity':identity['identity'],'request_id':'a'*32,'method':'ping'})
            self.assertEqual(_recv(sock)['error']['code'],'coordinator_auth')
        payload={'token':identity['token'],'identity':identity['identity'],'request_id':'b'*32,'method':'submit','source':"j=opt.jobs.submit('adam',options={'maxiter':2});"}
        replies=[]
        for _ in range(2):
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
                sock.connect(identity['socket']); _send(sock,payload); replies.append(_recv(sock))
        self.assertEqual(replies[0],replies[1])
        jobs=self.command('opt.jobs.status();'); self.assertEqual(jobs['jobs']['length'],1)

    def test_malformed_transport_and_cancelled_input_preserve_owner(self):
        import struct
        _,identity=_identity(self.path)
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
            sock.connect(identity['socket']); sock.sendall(struct.pack('!I',2)+b'[]')
            self.assertEqual(_recv(sock)['error']['code'],'coordinator_request_failed')
        self.assertEqual(request(self.path,'ping')['lifetime'],'coordinator')
        self.assertEqual(self.client.cancel_input('r=opt.jobs.'),130)
        rows=self.records()
        self.assertTrue(any(row['kind']=='submitted' and row['data']['origin']=='input_cancelled'
                            and row['data']['raw']=='r=opt.jobs.' for row in rows))
    def test_protected_records_capacity_refusal_cleanup_and_release(self):
        self.command("j=opt.jobs.submit('adam',options={'maxiter':50});opt.jobs.pause(j);opt.jobs.wait(j,timeout=10);")
        index=json.loads((self.path/'session.json').read_text())
        registry=index['jobs']; identity=next(iter(registry['records'].values()))['payload']
        self.assertTrue(index['objects'][identity]['available'])
        self.command('opt.storage.configure(max_bytes=20000000);',2)
        self.assertEqual(self.responses[-1].error['code'],'storage_full')
        self.command('plan=opt.storage.plan();')
        targets=self.responses[-1].metadata['summary'].get('targets',{})
        self.assertNotIn(identity,targets if isinstance(targets,list) else targets.get('items',[]))
        self.command('opt.jobs.cancel(j);opt.jobs.wait(j,timeout=10);opt.jobs.release(j);')
        self.assertEqual(json.loads((self.path/'session.json').read_text())['jobs']['records'],{})
    def test_cli_entry_survives_eof_and_reconnect(self):
        env=dict(os.environ,PYTHONPATH=str(ROOT))
        run=subprocess.run([sys.executable,'-B','-m','optimizer.cli','--connect',str(self.path),'--json','--command',"j=opt.jobs.submit('adam',options={'maxiter':6});"],cwd=ROOT,env=env,capture_output=True,text=True,timeout=20)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        self.command('opt.jobs.wait(j,timeout=10);r=opt.jobs.result(j);')
        self.assertTrue((self.path/'runtime/coordinator.json').exists())
    def test_fork_inherits_candidates_without_live_job_identity(self):
        self.command("j=opt.jobs.submit('adam',options={'maxiter':2});opt.jobs.wait(j,timeout=10);r=opt.jobs.result(j);")
        self.command(f"opt.project.fork('fork',root={str(self.root)!r});")
        fork=json.loads((self.root/'.optimizer/fork/session.json').read_text())
        self.assertNotIn('jobs',fork)

    def test_named_handles_survive_retention_pressure_until_release(self):
        self.command('opt.storage.configure(keep_recent=0,undo_depth=0);')
        self.command("a=opt.jobs.submit('adam',options={'maxiter':2});b=opt.jobs.submit('momentum',options={'maxiter':2});")
        self.command('opt.jobs.wait(a,timeout=10);opt.jobs.wait(b,timeout=10);')
        for _ in range(3): self.command('opt.jobs.status(a);opt.jobs.status(b);')
        self.assertEqual(self.command('opt.jobs.compare([a,b],metric="J");')['ranking']['length'],2)
        index=json.loads((self.path/'session.json').read_bytes())
        self.assertTrue(all(index['objects'][index['bindings'][name]]['available'] for name in ('a','b')))
        self.client.finish();self.client=_RemoteTerminal(self.path,emit=self.responses.append)
        self.command('opt.jobs.result(a);opt.jobs.release(a);opt.jobs.release(b);')
        self.assertEqual(json.loads((self.path/'session.json').read_bytes())['jobs']['records'],{})

    def test_owned_coordinator_crash_recovery_requires_explicit_physical_restart(self):
        self.stop()
        process=subprocess.Popen([sys.executable,'-B','-m','optimizer.jobs.coordinator',str(self.path),'exact'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            env=dict(os.environ,PYTHONPATH=str(ROOT)),start_new_session=True)
        self.addCleanup(lambda:process.wait(timeout=5) if process.poll() is not None else process.terminate())
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            try: request(self.path,'ping'); break
            except _ServiceError: time.sleep(.02)
        else: self.fail('Owned coordinator did not start')
        self.client=_RemoteTerminal(self.path,emit=self.responses.append)
        self.command("j=opt.jobs.submit('adam',options={'maxiter':40,'step_size':.02});opt.jobs.pause(j);opt.jobs.wait(j,timeout=10);")
        index=json.loads((self.path/'session.json').read_bytes())
        payload=next(iter(index['jobs']['records'].values()))['payload']
        from optimizer.checkpoints.codec import _Codec
        record=_Codec(lambda d:(self.path/f'checkpoints/blobs/{d}.npz').read_bytes()).decode(json.loads((self.path/f'checkpoints/manifests/{payload}.json').read_bytes())['payload'])['value']
        expected=opt.adam(QuadraticVectorSystem(N=5),record['pause_state']['physical'],maxiter=3,step_size=.02)
        _,old_identity=_identity(self.path)
        process.kill(); process.wait(timeout=5)
        with self.assertRaises(_ServiceError): request(self.path,'ping')
        start(self.path)
        self.assertFalse(Path(old_identity['socket']).exists())
        self.client=_RemoteTerminal(self.path,emit=self.responses.append)
        status=self.command('opt.jobs.status(j);')
        self.assertEqual(status['status'],'interrupted'); self.assertEqual(status['error']['code'],'coordinator_lost')
        self.command('opt.jobs.resume(j);',2)
        self.command("child=opt.jobs.resume(j,mode='physical',maxiter=3);opt.jobs.wait(child,timeout=10);r=opt.jobs.result(child);")
        self.assertEqual(self.command('r.metrics;'),expected.metrics)

    def test_multiple_clients_serialize_independent_submissions(self):
        from concurrent.futures import ThreadPoolExecutor
        def submit(index):
            return request(self.path,'submit',source=f"j{index}=opt.jobs.submit('adam',options={{'maxiter':3}},label='start-{index}');")['status']
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(submit,range(4))),[0]*4)
        self.command('opt.jobs.wait(j0,timeout=10);opt.jobs.wait(j1,timeout=10);opt.jobs.wait(j2,timeout=10);opt.jobs.wait(j3,timeout=10);')
        comparison=self.command("opt.jobs.compare([j0,j1,j2,j3],metric='J');")
        self.assertEqual(comparison['ranking']['length'],4)
        self.assertTrue(all(row['kind']!='submitted' or row['data']['raw']!='' or row['data']['origin']!='command' for row in self.records()))

class DurableFailureTests(unittest.TestCase):
    def test_failed_outcome_write_does_not_acknowledge_or_duplicate_completion(self):
        with tempfile.TemporaryDirectory() as root:
            terminal=_Terminal(); manager=terminal.service._jobs
            try:
                self.assertEqual(terminal.submit(f"opt.project.init('failure',root={root!r});opt.session.load('main',{str(SOURCE)!r},parameters={{'N':5,'delay':.05}},checkpoint={{'stateless':True,'dependencies':['quadratic_system.py']}});u=opt.guesses.constant_guess(value=.1);opt.session.use(u);"),0)
                manager.durable=True; manager.runtime_id=terminal.service._persistence.store.project['id']
                flush(terminal)
                self.assertEqual(terminal.submit("j=opt.jobs.submit('adam',options={'maxiter':2});"),0)
                handle=terminal.service.session.get('j')
                deadline=time.monotonic()+8
                while manager.records[handle.job_id].status not in {'succeeded','failed'} and time.monotonic()<deadline: time.sleep(.02)
                store=terminal.service._persistence.store; before=store.index['jobs']['signature']
                with patch.object(_Journal,'outcome',side_effect=OSError('fsync unavailable')):
                    with self.assertRaises(OSError): flush(terminal)
                self.assertEqual(store.index['jobs']['signature'],before)
                flush(terminal)
                count=len(list(_Journal(store).rows())); flush(terminal)
                self.assertEqual(len(list(_Journal(store).rows())),count)
                self.assertNotEqual(store.index['jobs']['signature'],before)
            finally:
                manager.durable=False; terminal.service.close()
