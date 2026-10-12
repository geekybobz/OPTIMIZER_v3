"""Native MCP transactions, shared owners and real official-SDK stdio clients."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
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
from optimizer.jobs.coordinator import start,request,_RemoteTerminal,_identity,_send,_recv
from optimizer.mcp.adapter import _Adapter
from optimizer.operations.frontend import dispatch
from optimizer.operations.contracts import _ServiceError
from fixtures.quadratic_system import QuadraticVectorSystem

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tests/fixtures/quadratic_system.py'
JOB=ROOT/'tests/fixtures/job_system.py'
SHOT=ROOT/'tests/fixtures/job_shooting.py'
SDK=importlib.util.find_spec('mcp') is not None

def ref(name,*path): return {'$ref':{'object_id':name,'path':list(path)}}

class TypedFrontendTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.terminal=_Terminal();self.addCleanup(self.terminal.service.close)
        self.call('project.init',{'name':'typed','root':self.temp.name})
    def call(self,operation,arguments=None,**kwargs):
        response=dispatch(self.terminal.history,'execute',{'operation':operation,'arguments':arguments or {},**kwargs})
        self.assertEqual(response['status'],'ok',response.get('error'));return response
    def load(self):
        self.call('session.load',{'name':'main','factory':str(SOURCE),'parameters':{'N':9},'checkpoint':{'stateless':True,'dependencies':[]}})
        self.call('guesses.constant_guess',{'value':.1},bind='u');self.call('session.use',{'value':ref('u')})
    def rows(self): return list(_Journal(self.terminal.service._persistence.store).rows())

    def test_typed_numerical_parity_and_truthful_history_without_cli_parser(self):
        with patch('optimizer.cli.language._parse',side_effect=AssertionError('CLI parsing forbidden')):
            self.load()
            result=self.call('optimizers.adam',{'maxiter':3,'step_size':.02},bind='r',activate=False)
        expected=opt.adam(QuadraticVectorSystem(N=9),opt.constant_guess(QuadraticVectorSystem(N=9),value=.1),maxiter=3,step_size=.02)
        np.testing.assert_array_equal(self.terminal.service.session.get('r').controls.u,expected.controls.u)
        head=self.terminal.service.session.active();np.testing.assert_array_equal(head.u,np.full((3,9),.1))
        submitted=[row for row in self.rows() if row['kind']=='submitted']
        self.assertTrue(all(row['data']['origin']=='mcp' and row['data']['raw']=='' for row in submitted))
        self.assertEqual(submitted[-1]['data']['expression']['request']['operation'],'optimizers.adam')
        self.assertEqual(result['metadata']['summary']['metrics'],expected.metrics)

    def test_discovery_session_lifecycle_and_unsupported_entries_are_explicit(self):
        listed=dispatch(self.terminal.history,'list',{'group':'session','limit':50})
        self.assertIn('session.load',[row['id'] for row in listed['metadata']['summary']['operations']])
        supported=dispatch(self.terminal.history,'info',{'operation':'optimizers.adam'})
        self.assertTrue(supported['metadata']['summary']['transport']['supported'])
        unsupported=dispatch(self.terminal.history,'info',{'operation':'session.attach'})
        self.assertFalse(unsupported['metadata']['summary']['transport']['supported'])
        bad=dispatch(self.terminal.history,'execute',{'operation':'session.attach','arguments':{'system':'pretend'}})
        self.assertEqual(bad['error']['code'],'adapter_unavailable')

    def test_revision_identity_invalid_envelope_and_failed_solver_leave_head(self):
        self.load();before=self.terminal.service.session.snapshot()
        for extra,code in [({'expected_revision':before['revision']-1},'revision_conflict'),({'experiment_id':'wrong'},'experiment_changed')]:
            response=dispatch(self.terminal.history,'execute',{'operation':'optimizers.adam','arguments':{'maxiter':1},**extra})
            self.assertEqual(response['error']['code'],code)
        for arguments in [{'maxiter':-1},{'maxiter':1,'controls':ref('missing')},{'maxiter':1,'controls':{'$import':'os'}}, {'maxiter':1,'step_size':float('nan')}]:
            response=dispatch(self.terminal.history,'execute',{'operation':'optimizers.adam','arguments':arguments})
            self.assertEqual(response['status'],'error')
        self.assertEqual(self.terminal.service.session.snapshot(),before)
        self.assertEqual(sum(row['kind']=='submitted' for row in self.rows()),sum(row['kind']=='outcome' for row in self.rows()))

    def test_inspection_pages_public_fields_and_input_budgets(self):
        self.load();self.call('optimizers.adam',{'maxiter':1},bind='r',activate=False)
        view=dispatch(self.terminal.history,'inspect',{'reference':{'object_id':'r','path':['controls']},'detail':'values','offset':2,'limit':4})
        self.assertEqual(view['metadata']['summary']['items'],self.terminal.service.session.get('r').controls.u.reshape(-1)[2:6].tolist())
        bad=dispatch(self.terminal.history,'inspect',{'reference':{'object_id':'r','path':['_private']}})
        self.assertEqual(bad['status'],'error')
        huge=dispatch(self.terminal.history,'execute',{'operation':'session.state','arguments':{'x':'x'*70000}})
        self.assertEqual(huge['error']['code'],'input_limit')
        for field in ('operation','request_id'):
            rejected=dispatch(self.terminal.history,'execute',{'operation':'session.state',field:'x'*65000})
            self.assertEqual(rejected['error']['code'],'input_limit')
            self.assertLessEqual(len(json.dumps(rejected).encode()),65536)

    def test_control_conversion_smoothing_filtering_and_explicit_activation(self):
        self.load();self.call('session.load',{'name':'fine','factory':str(SOURCE),'parameters':{'N':17},'checkpoint':{'stateless':True,'dependencies':[]}})
        self.call('base.control_spec',{},context='fine',bind='target')
        for name,count in [('g0',9),('g1',17)]:
            self.call('utils.uniform_control_grid',{'start':0,'stop':1,'count':count,'placement':'nodes'},bind=name)
        self.call('utils.resample_controls',{'controls':ref('u'),'source_grid':ref('g0'),'target_grid':ref('g1'),'target':ref('target')},bind='converted')
        self.call('utils.smooth_controls',{'controls':ref('converted','controls'),'sigma':1,'boundary':'reflect'},bind='smooth')
        self.call('utils.filter_controls',{'controls':ref('smooth','controls'),'grid':ref('g1'),'kind':'lowpass','cutoff':3,'boundary':'periodic'},bind='filtered')
        self.call('utils.scale_controls',{'controls':ref('filtered','controls'),'factors':2},bind='scaled')
        self.call('session.use',{'value':ref('scaled','controls')})
        self.assertEqual(self.terminal.service.session.active().u.shape,(3,17))
        np.testing.assert_allclose(self.terminal.service.session.active().u,.2)

    def test_olss_native_solver_and_qualification_parity(self):
        self.call('session.load',{'name':'shoot','factory':str(SHOT),'parameters':{'delay':0},'checkpoint':{'stateless':True,'dependencies':['shooting_systems.py']}})
        self.call('system_olss.validate_shooting_spec',{},bind='spec')
        self.call('system_olss.ShootingVariables',{'spec':ref('spec'),'values':[.4]},bind='v')
        result=self.call('shooting.solve_shooting',{'initial':ref('v'),'maxiter':20},bind='r',activate=False)
        value=self.terminal.service.session.get('r')
        from fixtures.shooting_systems import CubicRootShootingSystem
        system=CubicRootShootingSystem();initial=opt.ShootingVariables(system.shooting_spec(),np.array([.4]))
        expected=opt.solve_shooting(system,initial,maxiter=20)
        np.testing.assert_array_equal(value.variables.values,expected.variables.values)
        self.assertEqual(value.verified,expected.verified);self.assertEqual(value.qualified,expected.qualified)
        self.assertEqual(result['metadata']['summary']['metrics'],expected.metrics)

    def test_checkpoint_undo_redo_and_typed_command_replay_refusal(self):
        self.load();self.call('session.save',{'name':'start','mode':'controls_only'})
        self.call('optimizers.adam',{'maxiter':2});self.call('session.undo');self.call('session.redo')
        self.call('session.restore',{'checkpoint':'start','mode':'controls_only'})
        np.testing.assert_allclose(self.terminal.service.session.active().u,.1)
        plan=self.terminal.history.replay_plan(mode='commands')
        self.assertFalse(plan['executable'])
        self.assertTrue(any(issue['code']=='typed_history_requires_checkpoint' for issue in plan['issues']))
        self.call('optimizers.adam',{'maxiter':0})
        self.call('history.replay_plan',{'mode':'checkpoint'},bind='plan')
        self.call('history.replay',{'plan':ref('plan'),'project':'restored','root':self.temp.name})
        restored=json.loads((Path(self.temp.name)/'.optimizer/restored/session.json').read_bytes())
        self.assertEqual(restored['contexts']['main']['family'],'olgs');self.assertNotIn('jobs',restored)

    def test_failed_history_write_does_not_publish_or_launch(self):
        self.load();before=self.terminal.service.session.snapshot()
        with patch.object(_Journal,'outcome',side_effect=OSError('fsync failed')):
            result=dispatch(self.terminal.history,'execute',{'operation':'optimizers.adam','arguments':{'maxiter':2},'bind':'failed'})
        self.assertEqual(result['status'],'error');self.assertEqual(self.terminal.service.session.snapshot(),before)
        self.call('session.state')
        self.assertFalse(_Journal(self.terminal.service._persistence.store).pending)

    def test_failed_submission_history_never_starts_worker(self):
        self.load()
        with patch.object(_Journal,'outcome',side_effect=OSError('fsync failed')),patch('optimizer.jobs.manager.subprocess.Popen') as launch:
            result=dispatch(self.terminal.history,'execute',{'operation':'jobs.submit','arguments':{'operation':'adam','options':{'maxiter':2}},'bind':'j'})
        self.assertEqual(result['status'],'error');launch.assert_not_called()
        self.assertEqual(self.terminal.service._jobs.records,{})

    def test_tuple_constructor_and_schedule_references(self):
        self.call('system_olss.ShootingBlock',{'name':'x'},bind='block')
        self.call('system_olss.ShootingSpec',{'blocks':{'$tuple':[ref('block')]}},bind='spec')
        self.call('system_olss.ShootingVariables',{'spec':ref('spec'),'values':[.3]},bind='v')
        self.assertEqual(self.terminal.service.session.get('v').values.tolist(),[.3])
        self.load();self.call('schedules.ConstantSchedule',{'value':.02},bind='step')
        self.call('optimizers.adam',{'maxiter':2,'step_size':ref('step')},bind='r')
        self.assertEqual(self.terminal.service.session.get('r').iterations,2)

    def test_storage_quota_increase_uses_native_request(self):
        store=self.terminal.service._persistence.store
        store.index['policy']['max_bytes']=store.usage()+1000
        with patch('optimizer.cli.language._parse',side_effect=AssertionError('No CLI parser')):
            self.call('storage.configure',{'max_bytes':10_000_000})
        self.assertEqual(store.index['policy']['max_bytes'],10_000_000)

class _CoordinatorBase(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path=self.root/'.optimizer/shared'
        terminal=_Terminal()
        try:
            self.assertEqual(terminal.submit(f"opt.project.init('shared',root={str(self.root)!r});opt.session.load('main',{str(JOB)!r},parameters={{'N':5,'delay':.05}},checkpoint={{'stateless':True,'dependencies':['quadratic_system.py']}});u=opt.guesses.constant_guess(value=.1);opt.session.use(u);"),0)
            terminal.finish()
        finally:terminal.service.close()
        start(self.path);self.adapter=_Adapter();self.addCleanup(self.stop)
        self.adapter.attach(str(self.path))
    def stop(self):
        self.adapter.close()
        try:request(self.path,'submit',source='opt.jobs.shutdown();')
        except _ServiceError:pass
        deadline=time.monotonic()+5
        while (self.path/'runtime/coordinator.json').exists() and time.monotonic()<deadline:time.sleep(.02)
    def call(self,operation,arguments=None,**kwargs):
        response=self.adapter.call('execute',{'operation':operation,'arguments':arguments or {},**kwargs})
        self.assertEqual(response['status'],'ok',response.get('error'));return response

class SharedCoordinatorTests(_CoordinatorBase):
    def test_mixed_cli_mcp_jobs_pause_reconnect_and_result_selection(self):
        self.call('jobs.submit',{'operation':'adam','options':{'maxiter':12}},bind='j')
        self.call('jobs.pause',{'job':ref('j')});self.call('jobs.wait',{'job':ref('j'),'timeout':10})
        self.call('jobs.checkpoint',{'job':ref('j')},bind='boundary')
        self.adapter.detach();self.adapter.attach(str(self.path))
        cli=_RemoteTerminal(self.path,emit=lambda response:None)
        self.assertEqual(cli.submit('opt.jobs.resume(j);opt.jobs.wait(j,timeout=10);r=opt.jobs.result(j);'),0);cli.finish()
        result=self.adapter.call('inspect',{'reference':{'object_id':'r','path':['metrics']},'detail':'values'})
        self.assertEqual(result['status'],'ok')
        self.call('session.use',{'value':ref('r','controls')})
        self.call('jobs.release',{'job':ref('j')})
        rows=[json.loads(line) for line in (self.path/'history.jsonl').read_text().splitlines()]
        origins={row['data'].get('origin') for row in rows if row['kind']=='submitted'}
        self.assertTrue({'command','mcp','coordinator'}<=origins)

    def test_request_identity_conflict_cannot_repeat_different_work(self):
        import socket
        _,identity=_identity(self.path)
        base={'token':identity['token'],'identity':identity['identity'],'request_id':'f'*32,'method':'native','action':'execute','payload':{'operation':'guesses.constant_guess','arguments':{'value':.1},'bind':'first'}}
        replies=[]
        for value in [base,base,{**base,'payload':{'operation':'guesses.constant_guess','arguments':{'value':.2},'bind':'second'}}]:
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
                sock.connect(identity['socket']);_send(sock,value);replies.append(_recv(sock))
        self.assertEqual(replies[0],replies[1]);self.assertEqual(replies[2]['error']['code'],'request_conflict')

@unittest.skipUnless(SDK,'Install the mcp extra to run real MCP protocol acceptance tests.')
class MCPProtocolTests(_CoordinatorBase):
    def test_official_client_stdio_discovery_execution_errors_and_eof_detach(self):
        import anyio
        from mcp import ClientSession,StdioServerParameters
        from mcp.client.stdio import stdio_client
        async def exercise():
            params=StdioServerParameters(command=sys.executable,args=['-B','-m','optimizer.mcp','--experiment',str(self.path)],
                cwd=str(ROOT),env={**os.environ,'PYTHONPATH':str(ROOT),'PYTHONDONTWRITEBYTECODE':'1'})
            with tempfile.TemporaryFile(mode='w+') as errors:
                async with stdio_client(params,errlog=errors) as (read,write):
                    async with ClientSession(read,write) as client:
                        await client.initialize()
                        listing=await client.list_tools();self.assertEqual(len(listing.tools),6)
                        info=await client.call_tool('optimizer_info',{'operation':'optimizers.adam'})
                        self.assertFalse(info.isError)
                        result=await client.call_tool('optimizer_execute',{'operation':'optimizers.adam','arguments':{'maxiter':2},'bind':'wire','activate':False})
                        self.assertFalse(result.isError)
                        inspect=await client.call_tool('optimizer_inspect',{'reference':{'object_id':'wire','path':['metrics']},'detail':'values'})
                        self.assertFalse(inspect.isError);self.assertIn('J',inspect.structuredContent['metadata']['summary']['items'])
                        system=QuadraticVectorSystem(N=5)
                        expected=opt.adam(system,opt.constant_guess(system,value=.1),maxiter=2)
                        self.assertEqual(inspect.structuredContent['metadata']['summary']['items'],expected.metrics)
                        loaded=await client.call_tool('optimizer_execute',{'operation':'session.load','arguments':{'name':'wire_shoot','factory':str(SHOT),'parameters':{'delay':0},'checkpoint':{'stateless':True,'dependencies':['shooting_systems.py']}}})
                        self.assertFalse(loaded.isError)
                        await client.call_tool('optimizer_execute',{'operation':'system_olss.validate_shooting_spec','bind':'wire_spec'})
                        await client.call_tool('optimizer_execute',{'operation':'system_olss.ShootingVariables','arguments':{'spec':ref('wire_spec'),'values':[.4]},'bind':'wire_v'})
                        shot=await client.call_tool('optimizer_execute',{'operation':'shooting.solve_shooting','arguments':{'initial':ref('wire_v'),'maxiter':20},'bind':'wire_root','activate':False})
                        self.assertFalse(shot.isError);self.assertTrue(shot.structuredContent['metadata']['summary']['verified'])
                        await client.call_tool('optimizer_execute',{'operation':'session.select','arguments':{'name':'main'}})
                        bad=await client.call_tool('optimizer_execute',{'operation':'optimizers.adam','arguments':{'maxiter':-1}})
                        self.assertTrue(bad.isError);self.assertEqual(bad.structuredContent['status'],'error')
                        schema=await client.call_tool('optimizer_execute',{'operation':'optimizers.adam','unknown':1})
                        self.assertTrue(schema.isError)
                        unsupported=await client.call_tool('optimizer_info',{'operation':'session.attach'})
                        self.assertFalse(unsupported.structuredContent['metadata']['summary']['transport']['supported'])
                        over=await client.call_tool('optimizer_info',{'operation':'x'*70000})
                        self.assertTrue(over.isError);self.assertEqual(over.structuredContent['error']['code'],'input_limit')
                        detached=await client.call_tool('optimizer_detach',{});self.assertFalse(detached.isError)
                        invalid_path=await client.call_tool('optimizer_attach',{'path':'x'*70000})
                        self.assertTrue(invalid_path.isError)
                        self.assertLessEqual(len(invalid_path.structuredContent['error']['message']),2048)
                        attached=await client.call_tool('optimizer_attach',{'path':str(self.path)});self.assertFalse(attached.isError)
                errors.seek(0);self.assertEqual(errors.read(),'')
        anyio.run(exercise)
        self.assertEqual(request(self.path,'ping')['lifetime'],'coordinator')
        self.assertEqual(self.adapter.call('inspect',{'reference':{'object_id':'wire'}})['status'],'ok')

    def test_optional_cli_reports_missing_dependency_cleanly(self):
        # Simulate an SDK import failure without changing any environment.
        from optimizer.mcp.server import main
        real=__import__('builtins').__import__
        def imports(name,*args,**kwargs):
            if name=='mcp':raise ImportError('test')
            return real(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=imports),patch('sys.stderr'):
            self.assertEqual(main([]),2)

    def test_oversized_stdio_frame_terminates_without_processing_tool(self):
        run=subprocess.run([sys.executable,'-B','-m','optimizer.mcp'],cwd=ROOT,
            env=dict(os.environ,PYTHONPATH=str(ROOT)),input=b'x'*131073+b'\n',capture_output=True,timeout=10)
        self.assertEqual(run.returncode,0,run.stderr.decode())
        for line in run.stdout.splitlines():
            self.assertLessEqual(len(line),262144);self.assertEqual(json.loads(line)['jsonrpc'],'2.0')
