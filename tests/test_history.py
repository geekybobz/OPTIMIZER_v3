"""Statement atomicity, frontend ownership, durable recovery and replay."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from optimizer.cli.language import _Terminal
from optimizer.history.journal import _Journal
from optimizer.operations import OperationService, OperationRequest
from optimizer.persistence.store import _Store

FIXTURE=Path(__file__).parent/'fixtures/quadratic_system.py'
ROOT=Path(__file__).resolve().parents[1]

class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=self.temp.name
        self.responses=[];self.terminal=_Terminal(emit=self.responses.append)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.terminal.service._persistence.close)
    def run_command(self,source,status=0):
        self.assertEqual(self.terminal.submit(source),status,self.responses[-1].error if self.responses else None)
    def initialize(self):
        self.run_command(f"opt.project.init('baseline', root={self.root!r});")
    def system(self):
        self.run_command(f"opt.session.load('main', {str(FIXTURE)!r}, parameters={{'N': 9}}, checkpoint={{'stateless': True, 'dependencies': []}});")
        self.run_command("u = opt.constant(value=0.1); opt.session.use(u);")
    def journal(self):return _Journal(self.terminal.service._persistence.store)
    def test_one_pair_per_statement_and_all_outcomes(self):
        self.initialize();self.system()
        self.run_command('a = opt.adam(maxiter=2, step_size=0.02); a.metrics;')
        self.run_command('opt.adam(unrecognized=1);',2)
        self.run_command('opt.adam(;',2)
        self.run_command("opt.session.save('saved'); opt.adam(maxiter=1, step_size=0.02); opt.session.undo(); opt.session.redo(); opt.session.restore('saved');")
        rows=list(self.journal().rows());submissions=[r for r in rows if r['kind']=='submitted'];outcomes=[r for r in rows if r['kind']=='outcome']
        self.assertEqual(len(submissions),len(outcomes));self.assertEqual(len(submissions),13)
        self.assertEqual([r['command_id'] for r in submissions],[r['command_id'] for r in outcomes])
        self.assertEqual(sum(r['data']['status']=='error' for r in outcomes),2)
        self.assertTrue(any('undo' in r['data']['raw'] for r in submissions))
        successful=next(r for r in outcomes if any(e['operation']=='optimizers.adam' for e in r['data']['operations']))
        self.assertIn('metrics',successful['data']['summary']);self.assertIsNotNone(successful['data']['index'])
        self.assertEqual(successful['data']['operations'][0]['effective_arguments']['step_size'],0.02)
    def test_nested_constructor_parent_failure_rolls_back_every_child(self):
        self.initialize();self.system();before=self.terminal.service.session.snapshot();index=copy.deepcopy(self.journal().store.index)
        self.run_command('bad = opt.adam(step_size=opt.ConstantSchedule(0.1), maxiter=-1);',2)
        self.assertEqual(self.terminal.service.session.snapshot(),before)
        self.assertEqual(self.journal().store.index,index)
        pair=self.journal().commands(1);self.assertEqual(pair[0]['status'],'error')
        outcome=list(self.journal().rows())[-1]
        self.assertGreaterEqual(len(outcome['data']['operations']),2)
    def test_nested_success_commits_one_generation(self):
        self.initialize();self.system();generation=self.journal().store.index['generation']
        self.run_command('a = opt.adam(step_size=opt.ConstantSchedule(0.02), maxiter=2);')
        self.assertEqual(self.journal().store.index['generation'],generation+1)
        self.assertEqual(len(list(self.journal().rows())[-1]['data']['operations']),2)
    def test_interrupt_preserves_head_and_records_once(self):
        self.initialize();self.system();before=self.terminal.service.session.snapshot()
        with patch('optimizer.operations.service.OperationService._execute_tool',side_effect=KeyboardInterrupt):
            self.run_command('a = opt.adam(maxiter=4);',130)
        self.assertEqual(self.terminal.service.session.snapshot(),before)
        self.assertEqual(self.journal().commands(1)[0]['status'],'interrupted')
    def test_payload_failure_has_error_outcome_and_no_publication(self):
        self.initialize();self.system();before=self.terminal.service.session.snapshot()
        from optimizer.persistence import store as module
        atomic=module._atomic
        def fail(path,data):
            if 'manifests' in path.parts:raise OSError('injected preparation failure')
            return atomic(path,data)
        with patch.object(module,'_atomic',side_effect=fail):self.run_command('a = opt.adam(maxiter=1);',2)
        self.assertEqual(self.terminal.service.session.snapshot(),before)
        self.assertEqual(self.journal().commands(1)[0]['status'],'error')
    def test_index_failure_after_outcome_is_success_and_recovers(self):
        self.initialize();self.system()
        from optimizer.persistence import store as module
        atomic=module._atomic
        def fail(path,data):
            if path.name=='session.json':raise OSError('injected index failure')
            return atomic(path,data)
        with patch.object(module,'_atomic',side_effect=fail):self.run_command('a = opt.adam(maxiter=2);')
        expected=self.terminal.service.session.get('a').controls.as_matrix()
        path=str(self.journal().store.path)
        self.assertIn('index recovery',self.journal().store.warning)
        self.run_command('opt.project.close();');self.run_command(f'opt.project.open({path!r});')
        np.testing.assert_array_equal(self.terminal.service.session.get('a').controls.as_matrix(),expected)
    def test_intent_failure_never_computes(self):
        self.initialize();self.system();before=self.terminal.service.session.snapshot()
        with patch.object(_Journal,'begin',side_effect=OSError('intent fsync failed')),patch('optimizer.operations.service.OperationService._execute_tool') as numerical:
            self.run_command('opt.adam(maxiter=4);',2)
            numerical.assert_not_called()
        self.assertEqual(self.terminal.service.session.snapshot(),before)
    def test_outcome_failure_preserves_index_and_is_recovered_interrupted(self):
        self.initialize();self.system();before=self.terminal.service.session.snapshot();path=str(self.journal().store.path)
        original=_Journal.outcome
        def fail(self,*args,**kwargs):raise OSError('outcome fsync failed')
        with patch.object(_Journal,'outcome',fail):self.run_command('a = opt.adam(maxiter=1);',2)
        self.assertEqual(self.terminal.service.session.snapshot(),before)
        self.terminal.service._persistence.close()
        self.run_command(f'opt.project.open({path!r});')
        self.assertTrue(any(row['status']=='interrupted' for row in self.journal().commands(50)))
        self.assertNotIn('a',self.terminal.service.session.snapshot()['bindings'])
    def test_migration_is_explicit_and_keeps_ids(self):
        service=OperationService();created=service.execute(OperationRequest('project.init',{'name':'python','root':self.root}));self.assertTrue(created.ok)
        self.assertTrue(service.execute(OperationRequest('session.retain',{'name':'x','value':3})).ok)
        old=service.session.snapshot()['bindings']['x'];path=str(service._persistence.store.path);service._persistence.close()
        self.run_command(f'opt.project.open({path!r});',2)
        self.assertFalse((Path(path)/'history.jsonl').exists())
        self.run_command(f'opt.project.open({path!r}, migrate=True);')
        self.assertEqual(self.terminal.service.session.snapshot()['bindings']['x'],old)
        first=list(self.journal().rows())[0];self.assertEqual(first['data']['origin']['migration'],'index/1')
    def test_ordinary_python_is_unlogged_and_history_owned_open_is_read_only(self):
        from fixtures.quadratic_system import system
        import optimizer as opt
        prior=set(Path(self.root).iterdir());s=system();original_cwd=Path.cwd()
        try:
            os.chdir(self.root)
            result=opt.adam(s,opt.Controls.constant(s.control_spec(),0.1),maxiter=2)
        finally:os.chdir(original_cwd)
        self.assertEqual(set(Path(self.root).iterdir()),prior);self.assertFalse(hasattr(result,'trace'))
        service=OperationService();self.assertTrue(service.execute(OperationRequest('project.init',{'name':'python','root':self.root})).ok)
        path=service._persistence.store.path;self.assertFalse((path/'history.jsonl').exists());service._persistence.close()
        self.initialize();self.system();path=str(self.journal().store.path);self.run_command('opt.project.close();')
        denied=service.execute(OperationRequest('project.open',{'path':path}));self.assertFalse(denied.ok);self.assertEqual(denied.error['code'],'history_owned')
        self.assertTrue(service.execute(OperationRequest('project.open',{'path':path,'read_only':True})).ok)
        fork=service.execute(OperationRequest('project.fork',{'name':'python_fork','root':self.root}));self.assertTrue(fork.ok,fork.error)
        self.assertFalse((service._persistence.store.path/'history.jsonl').exists());service._persistence.close()
    def test_fork_and_multiple_attachments_have_independent_history(self):
        self.initialize();self.system();source=self.journal().store.path
        self.run_command("opt.project.fork('branch');")
        branch=self.journal().store.path
        self.assertNotEqual(source,branch);self.assertEqual(list(self.journal().rows())[0]['data']['origin']['project_id'],json.loads((source/'project.json').read_text())['id'])
        sid=self.terminal.history.session_id;self.run_command('opt.project.close();')
        other=_Terminal();self.addCleanup(other.service._persistence.close)
        self.assertEqual(other.submit(f'opt.project.open({str(branch)!r}); opt.history.list();'),0)
        sessions={r['session_id'] for r in _Journal(other.service._persistence.store).rows() if r['kind']=='submitted'}
        self.assertIn(sid,sessions);self.assertIn(other.history.session_id,sessions)
    def test_writer_lease_conflict(self):
        self.initialize();other=_Terminal(emit=self.responses.append)
        self.assertEqual(other.submit(f'opt.project.open({str(self.journal().store.path)!r});'),2)
        self.assertEqual(self.responses[-1].error['code'],'project_locked')
    def test_checkpoint_replay_and_command_replay(self):
        self.initialize();self.system();self.run_command("a = opt.adam(maxiter=2, step_size=0.02); opt.session.save('saved');")
        expected=self.terminal.service.session.get('a').controls.as_matrix()
        self.run_command("preview = opt.history.replay_plan(target='saved'); opt.history.replay(plan=preview, project='restored');")
        self.run_command("commands = opt.history.replay_plan(target='saved', mode='commands'); opt.history.replay(plan=commands, project='recalculated');")
        for name in ('restored','recalculated'):
            with _open_readonly(Path(self.root)/'.optimizer'/name) as service:
                context=service.session._contexts['main'];value=service.session.get(context.head)
                np.testing.assert_array_equal(value.controls.as_matrix(),expected)
    def test_latest_preview_is_fixed_and_overwritten_bookmark_uses_current_save(self):
        self.initialize();self.system()
        self.run_command("a = opt.adam(maxiter=1); opt.session.save('saved'); a = opt.adam(maxiter=2); opt.session.save('saved');")
        expected=self.terminal.service.session.get('a').controls.as_matrix()
        self.run_command("preview = opt.history.replay_plan(); opt.history.replay(plan=preview, project='latest');")
        self.run_command("preview = opt.history.replay_plan(target='saved'); opt.history.replay(plan=preview, project='current_save');")
        for name in ('latest','current_save'):
            with _open_readonly(Path(self.root)/'.optimizer'/name) as service:
                value=service.session.get(service.session._contexts['main'].head)
                np.testing.assert_array_equal(value.controls.as_matrix(),expected)
    def test_changed_source_blocks_command_replay(self):
        self.initialize();fixture=Path(self.root)/'system.py';fixture.write_bytes(FIXTURE.read_bytes())
        self.run_command(f"opt.session.load('main', {str(fixture)!r}, checkpoint={{'stateless': True, 'dependencies': []}});")
        fixture.write_text(fixture.read_text()+'\n# changed source\n')
        plan=self.terminal.history.replay_plan(mode='commands')
        self.assertFalse(plan['executable']);self.assertTrue(any(x['code']=='source_changed' for x in plan['issues']))
    def test_missing_index_is_rebuilt_from_history_without_numerical_work(self):
        self.initialize();self.system();self.run_command('a = opt.adam(maxiter=1);')
        path=self.journal().store.path;expected=self.terminal.service.session.get('a').controls.as_matrix()
        self.terminal.service._persistence.close();(path/'session.json').unlink()
        with patch('optimizer.operations.service.OperationService._execute_tool') as numerical:
            self.run_command(f'opt.project.open({str(path)!r});');numerical.assert_not_called()
        np.testing.assert_array_equal(self.terminal.service.session.get('a').controls.as_matrix(),expected)
    def test_full_history_can_raise_its_own_quota_and_close_releases_lease(self):
        self.initialize();self.system();store=self.journal().store
        # Fill only nonprotected explicit input storage, without editing history.
        self.run_command(f'opt.storage.configure(max_bytes={store.usage()+180000});')
        remaining=store.index['policy']['max_bytes']-store.usage()
        (store.path/'inputs/filler').write_bytes(b'x'*(remaining-100))
        with patch('optimizer.operations.service.OperationService._execute_tool') as numerical:
            self.run_command('opt.adam(maxiter=4);',2);numerical.assert_not_called()
        self.run_command('opt.storage.configure(max_bytes=2000000);')
        self.assertEqual(store.index['policy']['max_bytes'],2000000)
        self.terminal.finish();self.assertTrue(store.lock.closed)
    def test_ephemeral_history_is_bounded_and_has_errors_inspections(self):
        for i in range(270):self.run_command(f'x = {i};')
        self.run_command('x; opt.missing();',2)
        self.assertLessEqual(len(self.terminal.history.memory),256)
        self.assertLessEqual(self.terminal.history.memory_bytes,1024*1024)
        self.assertEqual(self.terminal.history.list(1)['commands'][0]['status'],'error')
        self.assertFalse((Path(self.root)/'.optimizer').exists())
    def test_pruning_keeps_history_and_replay_reports_required_payload_loss(self):
        self.initialize();self.system();self.run_command('a = opt.adam(maxiter=1);')
        target=self.journal().commands(1)[0]['command_id']
        oid=self.terminal.service.session.snapshot()['contexts']['main']['head']
        self.run_command('a = opt.adam(maxiter=1);')
        history=self.journal().path.read_bytes()
        self.run_command(f"p = opt.storage.plan(action='delete', ids=[{oid!r}]); opt.storage.apply(plan=p);")
        self.assertTrue(self.journal().path.read_bytes().startswith(history))
        plan=self.terminal.history.replay_plan(target)
        self.assertFalse(plan['executable']);self.assertTrue(any(x['code']=='payload_pruned' and x['required'] for x in plan['issues']))
    def test_modified_replay_index_is_rejected_even_with_recomputed_hash(self):
        from optimizer.checkpoints.codec import _hash,_json
        self.initialize();self.system();self.run_command('a = opt.adam(maxiter=1);')
        plan=self.terminal.history.replay_plan();plan['index']['bindings']={}
        plan['hash']=_hash(_json({k:v for k,v in plan.items() if k!='hash'}))
        with self.assertRaisesRegex(Exception,'fabricated checkpoint'):
            self.terminal.history.replay(plan,'fabricated')
        self.assertFalse((Path(self.root)/'.optimizer/fabricated').exists())
    def test_read_only_history_inspections_do_not_append(self):
        self.initialize();self.system();path=self.journal().store.path
        self.run_command('opt.project.close();');history=(path/'history.jsonl').read_bytes()
        self.run_command(f'opt.project.open({str(path)!r}, read_only=True); opt.history.list();')
        self.run_command('opt.adam(maxiter=1);',2);self.terminal.finish()
        self.assertEqual((path/'history.jsonl').read_bytes(),history)
    def test_invalid_reference_options_are_recorded_as_errors(self):
        self.initialize();self.system()
        self.run_command('opt.adam(context=u);',2)
        self.run_command('opt.adam(activate=u);',2)
        self.assertEqual([r['status'] for r in self.journal().commands(2)],['error','error'])
    def test_large_state_names_keep_failure_record_within_reserved_capacity(self):
        self.initialize()
        for prefix in ('a','b'):
            self.run_command(prefix+'x'*40000+' = 1;')
        self.run_command('opt.adam(unknown=1);',2)
        from optimizer.checkpoints.codec import _json
        row=list(self.journal().rows())[-1]
        self.assertIn('sha256',row['data']['state_before'])
        self.assertLess(len(_json(row)),131072)
    def test_legacy_logging_fields_decode_as_numerical_state_only(self):
        from optimizer.checkpoints.codec import _Codec
        from fixtures.quadratic_system import system
        import optimizer as opt
        s=system();result=opt.adam(s,opt.Controls.constant(s.control_spec(),0.1),maxiter=1)
        encoder=_Codec();encoded=encoder.encode(result)
        def legacy(node):
            if isinstance(node,dict):
                name=node.get('$type');fields=node.get('fields',{}).get('$dict',[])
                if name in ('OptimizerResult','RunState','WarmStartState'):
                    fields[:]=[row for row in fields if row[0]!='diagnostics']
                    fields.extend([['trace_id','old'],['checkpoint_ids',{'$list':['old-id']}]] )
                    if name=='OptimizerResult':fields.extend([['trace',None],['tracking_failures',{'$list':[]}]])
                for child in node.values():legacy(child)
            elif isinstance(node,list):
                for child in node:legacy(child)
        legacy(encoded)
        decoded=_Codec(lambda digest:encoder.blobs[digest]).decode(encoded)
        np.testing.assert_array_equal(decoded.controls.as_matrix(),result.controls.as_matrix())
        self.assertEqual(decoded.diagnostics,{});self.assertFalse(hasattr(decoded.state,'trace_id'))
    def test_quota_rejection_happens_before_numerics(self):
        self.initialize();self.system();store=self.journal().store
        self.run_command(f'opt.storage.configure(max_bytes={store.usage()+230000});')
        raw='x = '+repr('a'*65000)+';'
        self.run_command(raw,2)
        self.assertLessEqual(store.usage(),store.index['policy']['max_bytes'])
    def test_corruption_and_unknown_versions_fail_without_numerical_replay(self):
        for suffix in ('truncated','version'):
            with self.subTest(suffix=suffix):
                terminal=_Terminal();self.assertEqual(terminal.submit(f"opt.project.init('{suffix}', root={self.root!r});"),0)
                path=terminal.service._persistence.store.path;terminal.service._persistence.close()
                journal=path/'history.jsonl';data=journal.read_bytes()
                journal.write_bytes(data[:-1] if suffix=='truncated' else data.replace(b'optimizer-history/1',b'optimizer-history/9',1))
                fresh=_Terminal(emit=self.responses.append)
                with patch('optimizer.operations.service.OperationService._execute_tool') as numerical:
                    self.assertEqual(fresh.submit(f'opt.project.open({str(path)!r});'),2);numerical.assert_not_called()
                self.assertEqual(self.responses[-1].error['code'],'history_corrupt')
    def test_process_death_before_outcome_and_after_outcome(self):
        for after in (False,True):
            name='after' if after else 'before'
            code=f'''import os
from optimizer.cli.language import _Terminal
from optimizer.persistence import store as module
terminal=_Terminal()
assert terminal.submit("opt.project.init('{name}', root={self.root!r});")==0
'''
            if after:
                code+='''atomic=module._atomic
def crash(path,data):
    if path.name=='session.json':os._exit(77)
    return atomic(path,data)
module._atomic=crash
terminal.submit('x = 4;')
'''
            else:
                code+='''def crash(service):os._exit(77)
terminal.history.run('x = 4;',None,crash)
'''
            result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,77,result.stderr)
            terminal=_Terminal();self.addCleanup(terminal.service._persistence.close)
            self.assertEqual(terminal.submit(f"opt.project.open({str(Path(self.root)/'.optimizer'/name)!r});"),0)
            snapshot=terminal.service.session.snapshot()
            self.assertEqual('x' in snapshot['bindings'],after)
            rows=_Journal(terminal.service._persistence.store).commands(50)
            self.assertEqual(rows[-2]['status'],'ok' if after else 'interrupted')

class _open_readonly:
    def __init__(self,path):self.path=path
    def __enter__(self):
        self.service=OperationService();r=self.service.execute(OperationRequest('project.open',{'path':str(self.path),'read_only':True}));assert r.ok,r.error;return self.service
    def __exit__(self,*args):self.service._persistence.close()
