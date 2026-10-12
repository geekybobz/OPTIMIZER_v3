"""Actual process isolation, bounded admission and frontend transaction safety."""
from __future__ import annotations
import dataclasses
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import optimizer as opt
from optimizer.cli.language import _Terminal
from optimizer.history.journal import _Journal
from optimizer.operations.contracts import _ServiceError
from optimizer.checkpoints.codec import _Codec
from fixtures.quadratic_system import QuadraticVectorSystem

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'tests/fixtures/job_system.py'
QUADRATIC=ROOT/'tests/fixtures/quadratic_system.py'
SHOOTING=ROOT/'tests/fixtures/shooting_systems.py'


class JobsTests(unittest.TestCase):
    def setUp(self):
        self.service=opt.OperationService()
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.service.close)

    def call(self, operation, args=None, **options):
        r=self.service.execute(opt.OperationRequest(operation,args or {},**options))
        self.assertTrue(r.ok,r.error); return r

    def reject(self,operation,args=None,code=None,**options):
        before=self.service.session.snapshot()
        response=self.service.execute(opt.OperationRequest(operation,args or {},**options))
        self.assertFalse(response.ok)
        if code: self.assertEqual(response.error['code'],code,response.error)
        self.assertEqual(self.service.session.snapshot(),before)
        return response

    def load(self, delay=0.0, marker=None, path=FIXTURE, name='main'):
        self.call('session.load',{'name':name,'factory':str(path),'parameters':{'N':5,'delay':delay,'marker':marker},
                                 'checkpoint':{'stateless':True,'dependencies':['quadratic_system.py']}})
        return self.call('guesses.constant_guess',{'value':.1},activate=True).result

    def submit(self, **options):
        return opt.jobs.submit('adam',options={'maxiter':2,'step_size':.02,'gradient_tolerance':None},service=self.service,**options)

    def done(self, job):
        status=opt.jobs.wait(job,timeout=10,service=self.service)
        self.assertIn(status['status'],{'succeeded','failed','cancelled','interrupted'},status)
        return status

    def test_parallel_native_parity_private_systems_and_thread_hints(self):
        marker=str(Path(self.temp.name)/'events.jsonl')
        initial=self.load(delay=.2,marker=marker)
        native_env=os.environ.get('OPENBLAS_NUM_THREADS')
        a,b=self.submit(label='a'),self.submit(label='b')
        self.assertEqual(self.done(a)['status'],'succeeded')
        self.assertEqual(self.done(b)['status'],'succeeded')
        left,right=opt.jobs.result(a,service=self.service),opt.jobs.result(b,service=self.service)
        expected=opt.adam(QuadraticVectorSystem(N=5),initial,maxiter=2,step_size=.02,gradient_tolerance=None)
        np.testing.assert_array_equal(left.controls.u,expected.controls.u)
        np.testing.assert_array_equal(right.controls.u,expected.controls.u)
        records=[json.loads(line) for line in Path(marker).read_text().splitlines()]
        pids={r['pid'] for r in records}
        self.assertEqual(len(pids),2); self.assertNotIn(os.getpid(),pids)
        self.assertTrue(all(r['threads']=='1' for r in records))
        self.assertEqual(os.environ.get('OPENBLAS_NUM_THREADS'),native_env)
        self.assertEqual(self.service.session._context('main').system.calls,0)
        left.controls.u[:]=999
        self.assertFalse(np.any(opt.jobs.result(a,service=self.service).controls.u==999))

    def test_five_starts_comparison_and_explicit_selection(self):
        self.load()
        guesses=[opt.constant_guess(QuadraticVectorSystem(N=5),value=x) for x in (.0,.05,.1,.15,.2)]
        jobs=[self.submit(controls=c,label=f'start-{i}') for i,c in enumerate(guesses)]
        head=self.service.session.snapshot()['contexts']['main']['head']
        for job in jobs: self.assertEqual(self.done(job)['status'],'succeeded')
        comparison=opt.jobs.compare(jobs,metric='J',direction='minimize',service=self.service)
        self.assertEqual(len(comparison['ranking']),5)
        self.assertEqual(self.service.session.snapshot()['contexts']['main']['head'],head)
        chosen=opt.jobs.result(comparison['ranking'][0],service=self.service)
        self.call('session.use',{'value':chosen.controls})
        np.testing.assert_array_equal(self.service.session.active().u,chosen.controls.u)

    def test_frozen_inputs_options_and_changed_foreground_configuration(self):
        initial=self.load(delay=.2)
        options={'maxiter':2,'step_size':.02,'gradient_tolerance':None}
        job=opt.jobs.submit('adam',controls=initial,options=options,service=self.service)
        initial.u[:]=900; options['step_size']=900
        self.call('session.update',{'updates':{'objective_weight':2}})
        self.assertEqual(self.done(job)['status'],'succeeded')
        result=self.call('jobs.result',{'job':job})
        expected=opt.adam(QuadraticVectorSystem(N=5),opt.constant_guess(QuadraticVectorSystem(N=5),value=.1),
                          maxiter=2,step_size=.02,gradient_tolerance=None)
        np.testing.assert_array_equal(result.result.controls.u,expected.controls.u)
        self.assertTrue(result.metadata['stale_configuration'])

    def test_frozen_source_and_declared_dependency(self):
        directory=Path(self.temp.name)
        for source in (FIXTURE,QUADRATIC): shutil.copy2(source,directory/source.name)
        initial=self.load(delay=.3,path=directory/FIXTURE.name)
        job=self.submit()
        path=directory/FIXTURE.name
        path.write_text(path.read_text().replace('SOURCE_OFFSET = 0.0','SOURCE_OFFSET = 900.0'))
        self.assertEqual(self.done(job)['status'],'succeeded')
        expected=opt.adam(QuadraticVectorSystem(N=5),initial,maxiter=2,step_size=.02,gradient_tolerance=None)
        np.testing.assert_array_equal(opt.jobs.result(job,service=self.service).controls.u,expected.controls.u)

    def test_source_recipe_and_option_validation_precedes_process_start(self):
        self.call('session.load',{'name':'bad','factory':str(QUADRATIC),'parameters':{'N':5}})
        self.call('guesses.constant_guess',activate=True)
        self.reject('jobs.submit',{'operation':'adam','options':{'maxiter':2}},'unsupported_factory')
        self.load(name='good')
        with patch('optimizer.jobs.manager.subprocess.Popen',side_effect=AssertionError('started')):
            for args in ({'operation':'adam','options':{'maxiterr':2}},
                         {'operation':'adam','options':{'maxiter':-1}},
                         {'operation':'utils.metric_report','options':{'maxiter':2}},
                         {'operation':'cma_es','options':{'maxiter':2}},
                         {'operation':'adam','options':{'maxiter':2,'parallel':True}}):
                self.reject('jobs.submit',args)
        self.assertEqual(self.service._jobs.records,{})

    def test_bounded_queue_and_rejected_submission_is_atomic(self):
        opt.jobs.configure(max_workers=1,max_queue=1,service=self.service)
        self.load(delay=.8)
        a,b=self.submit(),self.submit()
        self.reject('jobs.submit',{'operation':'adam','options':{'maxiter':2}},'queue_full')
        self.assertEqual(len(self.service._jobs.records),2)
        self.assertEqual(opt.jobs.status(b,service=self.service)['status'],'queued')
        opt.jobs.cancel(b,service=self.service)
        self.assertEqual(opt.jobs.status(b,service=self.service)['status'],'cancelled')
        opt.jobs.cancel(a,service=self.service); self.done(a)

    def test_memory_scratch_input_and_record_admission(self):
        self.load()
        self.reject('jobs.submit',{'operation':'adam','options':{'maxiter':2},'worker_bytes':2*1024**3},'memory_admission')
        opt.jobs.configure(scratch_bytes=4096,service=self.service)
        self.reject('jobs.submit',{'operation':'adam','options':{'maxiter':2}},'scratch_admission')
        opt.jobs.configure(scratch_bytes=128*1024**2,input_bytes=4096,service=self.service)
        self.reject('jobs.submit',{'operation':'adam','options':{'maxiter':2}},'input_quota')
        opt.jobs.configure(input_bytes=8*1024**2,max_jobs=1,service=self.service)
        job=self.submit(); self.done(job)
        self.reject('jobs.submit',{'operation':'adam','options':{'maxiter':2}},'job_quota')
        opt.jobs.release(job,service=self.service)
        next_job=self.submit(); self.done(next_job)

    def test_cancel_acknowledged_at_project_call_boundary(self):
        self.load(delay=.25)
        job=opt.jobs.submit('adam',options={'maxiter':100,'step_size':.01,'gradient_tolerance':None},service=self.service)
        deadline=time.monotonic()+5
        while opt.jobs.status(job,service=self.service)['status']=='queued' and time.monotonic()<deadline: time.sleep(.02)
        time.sleep(.4)
        opt.jobs.cancel(job,service=self.service)
        status=self.done(job)
        self.assertEqual(status['status'],'cancelled',status)
        self.assertEqual(status['cancellation_boundary'],'project_call')
        self.assertFalse(status['result_available'])
        opt.jobs.cancel(job,service=self.service)
        self.assertEqual(opt.jobs.status(job,service=self.service)['status'],'cancelled')

    def test_unresponsive_worker_forced_interruption_and_reaped(self):
        opt.jobs.configure(shutdown_seconds=.2,service=self.service)
        self.load(delay=5)
        job=self.submit()
        time.sleep(.5)
        opt.jobs.cancel(job,service=self.service)
        status=self.done(job)
        self.assertEqual(status['status'],'interrupted',status)
        self.assertTrue(status['forced']); self.assertFalse(status['result_available'])
        self.assertIsNotNone(self.service._jobs.records[job.job_id].process.poll())

    def test_sampled_worker_memory_limit_preserves_failure_reason(self):
        self.load(delay=.2)
        with patch('optimizer.jobs.manager._rss',return_value=1024**3):
            job=self.submit(); status=self.done(job)
        self.assertIn(status['status'],{'cancelled','interrupted'})
        self.assertEqual(status['error']['code'],'worker_memory_limit')

    def test_worker_exit_failure_and_result_byte_limit(self):
        self.call('session.load',{'name':'crash','factory':str(FIXTURE),'parameters':{'N':5,'crash':True},
                  'checkpoint':{'stateless':True,'dependencies':['quadratic_system.py']}})
        self.call('guesses.constant_guess',activate=True)
        job=self.submit(); self.assertEqual(self.done(job)['status'],'interrupted')
        self.reject('jobs.result',{'job':job},'job_failed')
        self.load(name='normal')
        job=self.submit(result_bytes=4096)
        status=self.done(job)
        self.assertEqual(status['status'],'failed',status)
        self.assertEqual(status['error']['code'],'result_quota')

    def test_worker_start_failure_after_publication_is_inspectable(self):
        self.load()
        with patch('optimizer.jobs.manager.threading.Thread.start',side_effect=RuntimeError('no thread')):
            job=self.submit()
        status=opt.jobs.status(job,service=self.service)
        self.assertEqual(status['status'],'failed')
        self.assertEqual(status['error']['code'],'worker_start_failed')
        self.reject('jobs.result',{'job':job},'job_failed')

    def test_lost_parent_worker_removes_private_transport(self):
        marker=str(Path(self.temp.name)/'orphan-marker.jsonl')
        script=f'''
import optimizer as opt, time, json, os
s=opt.OperationService()
opt.jobs.configure(shutdown_seconds=.2,service=s)
response=s.execute(opt.OperationRequest('session.load',{{'name':'main','factory':{str(FIXTURE)!r},'parameters':{{'N':5,'delay':5,'marker':{marker!r}}},'checkpoint':{{'stateless':True,'dependencies':['quadratic_system.py']}}}}))
assert response.ok, response.error
assert s.execute(opt.OperationRequest('guesses.constant_guess',activate=True)).ok
j=opt.jobs.submit('adam',options={{'maxiter':2}},service=s)
for i in range(300):
    if os.path.exists({marker!r}): break
    time.sleep(.02)
else: raise RuntimeError('worker did not start')
print(json.dumps({{'directory':str(s._jobs.records[j.job_id].directory)}}),flush=True)
os._exit(0)
'''
        response=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,text=True,capture_output=True,timeout=10)
        self.assertEqual(response.returncode,0,response.stderr)
        directory=Path(json.loads(response.stdout)['directory'])
        deadline=time.monotonic()+4
        while directory.exists() and time.monotonic()<deadline: time.sleep(.05)
        self.assertFalse(directory.exists(),'Lost-owner transport was not reclaimed')

    def test_shooting_without_independent_verification_is_not_ranked(self):
        self.call('session.load',{'name':'shot','factory':str(SHOOTING),'factory_name':'RectangularShootingSystem',
                                'checkpoint':{'stateless':True,'dependencies':[]}})
        self.call('session.use',{'value':opt.ShootingVariables(self.service.session._context('shot').layout,[0,0])})
        job=opt.jobs.submit('shooting.solve_shooting',options={'maxiter':3},service=self.service)
        self.assertEqual(self.done(job)['status'],'failed')
        result=opt.jobs.result(job,service=self.service)
        self.assertEqual(result.stop_reason,'verification_not_independent')
        self.assertEqual(opt.jobs.compare([job],metric='residual_norm',service=self.service)['ranking'],[])

    def test_result_not_ready_incompatible_comparison_and_no_auto_activation(self):
        self.load(delay=.3)
        job=self.submit()
        self.reject('jobs.result',{'job':job},'job_not_ready')
        self.done(job)
        head=self.service.session.snapshot()['contexts']['main']['head']
        response=self.call('jobs.result',{'job':job},bind='candidate')
        self.assertFalse(response.metadata['stale_configuration'])
        self.assertEqual(self.service.session.snapshot()['contexts']['main']['head'],head)
        self.call('session.update',{'updates':{'objective_weight':2}})
        other=self.submit(); self.done(other)
        self.reject('jobs.compare',{'jobs':[job,other]},'incompatible_comparison')
        self.reject('jobs.result',{'job':job},activate=True,code='invalid_arguments')

    def test_cma_seed_reproducibility(self):
        initial=self.load()
        options={'maxiter':1,'seed':73,'population_size':4}
        a=opt.jobs.submit('cma_es',options=options,service=self.service)
        b=opt.jobs.submit('cma_es',options=options,service=self.service)
        self.assertEqual(self.done(a)['status'],'succeeded'); self.assertEqual(self.done(b)['status'],'succeeded')
        np.testing.assert_array_equal(opt.jobs.result(a,service=self.service).controls.u,opt.jobs.result(b,service=self.service).controls.u)

    def test_olss_parity_verified_comparison_and_physical_selection(self):
        self.call('session.load',{'name':'shot','factory':str(SHOOTING),'factory_name':'CubicRootShootingSystem',
                                'checkpoint':{'stateless':True,'dependencies':[]}})
        spec=self.service.session._context('shot').layout
        variables=opt.ShootingVariables(spec,[.2])
        self.call('session.use',{'value':variables})
        a=opt.jobs.submit('shooting.solve_shooting',options={'maxiter':20,'tolerance':1e-8},service=self.service)
        self.assertEqual(self.done(a)['status'],'succeeded')
        result=opt.jobs.result(a,service=self.service)
        from fixtures.shooting_systems import CubicRootShootingSystem
        native=opt.solve_shooting(CubicRootShootingSystem(),variables,maxiter=20,tolerance=1e-8)
        np.testing.assert_array_equal(result.variables.values,native.variables.values)
        comparison=opt.jobs.compare([a],metric='residual_norm',service=self.service)
        self.assertEqual(comparison['rows'][0]['verified'],result.verified)
        self.assertEqual(comparison['ranking'],[a] if result.verified and result.qualified else [])

    def test_release_preserves_retained_candidate_and_unknown_id(self):
        self.load(); job=self.submit(); self.done(job)
        response=self.call('jobs.result',{'job':job},bind='candidate')
        opt.jobs.release(job,service=self.service)
        self.assertEqual(self.service._jobs.records,{})
        self.assertIsInstance(self.service.session.get('candidate'),opt.OptimizerResult)
        self.reject('jobs.status',{'job':job},'unknown_job')
        forged=dataclasses.replace(job,label='forged')
        self.reject('jobs.status',{'job':forged},'unknown_job')
        other=dataclasses.replace(job,runtime_id='previous-runtime')
        self.reject('jobs.status',{'job':other},'job_interrupted')

    def test_shutdown_config_validation_and_typed_handle_codec(self):
        for limits in ({'max_workers':True},{'max_workers':99999},{'max_queue':-1},{'shutdown_seconds':float('nan')},{'backend':'thread'}):
            self.reject('jobs.configure',limits)
        self.load(delay=.4); job=self.submit()
        self.reject('jobs.configure',{'max_workers':1},'jobs_active')
        self.reject('jobs.wait',{'job':job,'timeout':31},'invalid_arguments')
        codec=_Codec(); encoded=codec.encode(job)
        self.assertEqual(_Codec(codec.blobs.__getitem__).decode(encoded),job)
        opt.jobs.shutdown(service=self.service)
        self.assertIn(opt.jobs.status(job,service=self.service)['status'],{'cancelled','interrupted'})


class JobHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.responses=[]; self.terminal=_Terminal(emit=self.responses.append)
        self.addCleanup(self.terminal.service.close)

    def run_command(self, source, code=0):
        self.assertEqual(self.terminal.submit(source),code,self.responses[-1].error if self.responses else None)

    def initialize(self):
        self.run_command(f"opt.project.init('jobs',root={self.temp.name!r});")
        self.run_command(f"opt.session.load('main',{str(FIXTURE)!r},parameters={{'N':5,'delay':.1}},checkpoint={{'stateless':True,'dependencies':['quadratic_system.py']}});")
        self.run_command("u=opt.guesses.constant_guess(value=.1);opt.session.use(u);")

    def test_successful_history_before_start_and_failed_publication_never_starts(self):
        self.initialize()
        from optimizer.jobs.manager import _Jobs
        start=_Jobs._start
        observations=[]
        def observed(manager,job):
            journal=_Journal(self.terminal.service._persistence.store)
            outcomes=[row for row in journal.rows() if row['kind']=='outcome']
            observations.append(any(any(event['operation']=='jobs.submit' for event in row['data']['operations']) for row in outcomes))
            return start(manager,job)
        with patch.object(_Jobs,'_start',observed):
            self.run_command("j=opt.jobs.submit('adam',options={'maxiter':2});")
            self.run_command("s=opt.jobs.wait(j,timeout=10);")
        self.assertEqual(observations,[True])
        before=len(self.terminal.service._jobs.records)
        with patch.object(_Journal,'outcome',side_effect=OSError('outcome failed')):
            self.run_command("bad=opt.jobs.submit('adam',options={'maxiter':2});",2)
        self.assertEqual(len(self.terminal.service._jobs.records),before)

    def test_nested_statement_failure_discards_submission_and_cancellation(self):
        self.initialize()
        self.run_command("opt.utils.scale_controls(opt.jobs.submit('adam',options={'maxiter':2}),factors=2);",2)
        self.assertEqual(self.terminal.service._jobs.records,{})
        self.run_command("j=opt.jobs.submit('adam',options={'maxiter':100});")
        job=self.terminal.service.session.get('j')
        self.run_command("opt.utils.scale_controls(opt.jobs.cancel(j),factors=2);",2)
        self.assertFalse(self.terminal.service._jobs.records[job.job_id].cancel_requested)
        self.run_command("opt.jobs.cancel(j);opt.jobs.wait(j,timeout=10);")

    def test_result_checkpoint_reopen_lost_runtime_and_history_replay_boundary(self):
        self.initialize()
        self.run_command("j=opt.jobs.submit('adam',options={'maxiter':2,'step_size':.02});opt.jobs.wait(j,timeout=10);r=opt.jobs.result(j);r.metrics;")
        expected=self.terminal.service.session.get('r').controls.u.copy()
        self.run_command("saved=opt.checkpoints.save(r.controls,name='worker-control',kind='controls');")
        self.run_command("plan=opt.history.replay_plan(mode='commands');")
        plan=self.terminal.service.session.get('plan')
        self.assertTrue(any(issue['code']=='asynchronous_replay_requires_checkpoint' and issue['required'] for issue in plan['issues']))
        path=Path(self.temp.name)/'.optimizer/jobs'
        rows=list(_Journal(self.terminal.service._persistence.store).rows())
        self.assertTrue(any(row['kind']=='submitted' and 'jobs.result' in row['data']['raw'] for row in rows))
        self.run_command('opt.project.close();')
        other_responses=[]; other=_Terminal(emit=other_responses.append); self.addCleanup(other.service.close)
        self.assertEqual(other.submit(f'opt.project.open({str(path)!r});'),0,other_responses[-1].error)
        np.testing.assert_array_equal(other.service.session.get('r').controls.u,expected)
        self.assertEqual(other.submit("opt.session.restore('worker-control',mode='controls_only');"),0,other_responses[-1].error)
        np.testing.assert_array_equal(other.service.session.active().u,expected)
        self.assertEqual(other.submit('continued=opt.adam(maxiter=1);'),0,other_responses[-1].error)
        self.assertEqual(other_responses[-1].metadata['continuation'],'physical_restart')
        self.assertEqual(other.submit('opt.jobs.status(j);'),2)
        self.assertEqual(other_responses[-1].error['code'],'job_interrupted')

    def test_cli_script_and_eof_reap_workers(self):
        directory=Path(self.temp.name)
        source=(f"opt.project.init('cli',root={str(directory)!r});"
                f"opt.session.load('main',{str(FIXTURE)!r},parameters={{'N':5}},checkpoint={{'stateless':True,'dependencies':['quadratic_system.py']}});"
                "u=opt.guesses.constant_guess(value=.1);opt.session.use(u);"
                "a=opt.jobs.submit('adam',options={'maxiter':2});b=opt.jobs.submit('momentum',options={'maxiter':2});"
                "opt.jobs.wait(a,timeout=10);opt.jobs.wait(b,timeout=10);"
                "r=opt.jobs.result(a);opt.session.use(r.controls);r.metrics;")
        response=subprocess.run([sys.executable,'-B','-m','optimizer.cli','--json','--command',source],cwd=ROOT,
                                text=True,capture_output=True,timeout=25)
        self.assertEqual(response.returncode,0,response.stdout+response.stderr)
        rows=[json.loads(line) for line in response.stdout.splitlines()]
        self.assertIn('J',rows[-1]['metadata']['summary'])
        journal=json.loads((directory/'.optimizer/cli/history.jsonl').read_text().splitlines()[-1])
        self.assertEqual(journal['kind'],'outcome')


if __name__=='__main__': unittest.main()
