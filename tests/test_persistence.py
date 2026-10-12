"""Neutral persistence workflows, integrity, rollback, quotas and failure boundaries."""
from __future__ import annotations
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import optimizer as opt
from optimizer.checkpoints.codec import _Codec, _json
from optimizer.cli.language import _Terminal
from optimizer.persistence import store as store_module

_ROOT=Path(__file__).resolve().parents[1]
_SOURCE=_ROOT/'tests/fixtures/quadratic_system.py'
_SHOOT=_ROOT/'tests/fixtures/shooting_systems.py'

class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.service=opt.OperationService()
        self.services=[self.service]
    def tearDown(self):
        for service in self.services:
            service._persistence.close()
        self.temp.cleanup()
    def call(self,operation,args=None,service=None,**options):
        service=service or self.service
        response=service.execute(opt.OperationRequest(operation,args or {},**options))
        self.assertTrue(response.ok,response.error)
        return response
    def reject(self,operation,args=None,code=None,**options):
        before=self.service.session.snapshot()
        store=self.service._persistence.store
        generation=store.index['generation'] if store else None
        response=self.service.execute(opt.OperationRequest(operation,args or {},**options))
        self.assertFalse(response.ok)
        if code: self.assertEqual(response.error['code'],code,response.error)
        self.assertEqual(before,self.service.session.snapshot())
        if generation is not None: self.assertEqual(generation,store.index['generation'])
        return response
    def initialize(self,name='main',**policy):
        return self.call('project.init',dict(name=name,root=str(self.root),**policy))
    def load(self,checkpoint=True,**options):
        arguments={'name':'system','factory':str(_SOURCE),'parameters':{'N':9},**options}
        if checkpoint: arguments['checkpoint']={'stateless':True,'dependencies':[]}
        self.call('session.load',arguments)
        self.call('guesses.constant_guess',{'value':0.1},bind='u',activate=True)
    def adam(self,count=2,**options):
        return self.call('adam',{'maxiter':count,'step_size':0.01,'gradient_tolerance':None},bind=options.pop('bind','r'),**options)
    def reopen(self,mode='exact'):
        path=self.service._persistence.store.path
        self.call('project.close')
        self.call('project.open',{'path':str(path),'mode':mode})
    def save(self,name='baseline'):
        return self.call('session.save',{'name':name},bind=name)

    def test_exact_adam_reopen_and_split_parity(self):
        self.initialize(); self.load(); self.adam(); self.save()
        self.reopen(); self.adam()
        system=self.service.session._context(None).system
        native=opt.adam(system,self.service.session.get('u'),maxiter=4,step_size=0.01,gradient_tolerance=None)
        actual=self.service.session.get('r')
        np.testing.assert_array_equal(actual.controls.as_matrix(),native.controls.as_matrix())
        self.assertEqual(actual.state.global_iteration,4)
        self.assertFalse(hasattr(actual, "trace"))
        self.call('session.restore',{'checkpoint':'baseline'})
        self.assertEqual(self.service.session.active().state.global_iteration,2)
    def test_momentum_exact_state_parity(self):
        self.initialize(); self.load()
        args={'maxiter':2,'step_size':0.01,'gradient_tolerance':None}
        self.call('momentum',args,bind='r'); self.save(); self.reopen()
        self.call('momentum',args,bind='r')
        native=opt.momentum(self.service.session._context(None).system,self.service.session.get('u'),
                            maxiter=4,step_size=0.01,gradient_tolerance=None)
        np.testing.assert_array_equal(self.service.session.active().controls.as_matrix(),native.controls.as_matrix())
    def test_shooting_exact_continuation(self):
        self.initialize()
        self.call('session.load',{'name':'shoot','factory':str(_SHOOT),'factory_name':'RectangularShootingSystem',
                                 'checkpoint':{'stateless':True,'dependencies':[]}})
        system=self.service.session._context(None).system
        variables=opt.ShootingVariables(system.shooting_spec(),[0,0])
        ref=self.service.register_reference('v',variables)
        self.call('session.use',{'value':ref})
        self.call('solve_shooting',{'maxiter':1,'damping':1.0},bind='r')
        self.save(); self.reopen()
        result=self.call('solve_shooting',{'maxiter':1,'damping':1.0},bind='r')
        native=opt.solve_shooting(system,variables,maxiter=2,damping=1.0)
        np.testing.assert_array_equal(result.result.variables.values,native.variables.values)
        first=opt.solve_shooting(system,variables,maxiter=1,damping=1.0)
        split=opt.solve_shooting(system,first.variables,maxiter=1,damping=1.0,previous_result=first)
        self.assertEqual(result.result.history,split.history)
        # Native shooting records an extra boundary evaluation on continuation.
    def test_independent_projects_and_fork(self):
        self.initialize(); self.load(); self.adam(); self.save()
        original=self.service._persistence.store.path
        self.call('project.fork',{'name':'fork'})
        self.adam(); self.call('project.close')
        self.call('project.open',{'path':str(original)})
        self.assertEqual(self.service.session.active().state.global_iteration,2)
        self.assertEqual(len(self.call('project.list',{'root':str(self.root)}).result['projects']),2)
    def test_project_collision_unsafe_name_and_unsaved_open(self):
        self.initialize(); self.call('project.close')
        self.reject('project.init',{'name':'main','root':str(self.root)},'project_exists')
        self.reject('project.init',{'name':'../escape','root':str(self.root)},'invalid_arguments')
        self.service.register_reference('unsaved',1)
        self.reject('project.open',{'path':str(self.root/'.optimizer/main')},'unsaved_session')
    def test_undo_redo_restore_and_new_branch(self):
        self.initialize(); self.load(); self.adam(); self.save(); self.adam()
        bindings=dict(self.service.session.snapshot()['bindings'])
        self.call('session.undo'); self.assertEqual(self.service.session.active().state.global_iteration,2)
        self.assertEqual(bindings,self.service.session.snapshot()['bindings'])
        self.call('session.redo'); self.assertEqual(self.service.session.active().state.global_iteration,4)
        self.call('session.undo'); self.adam(1)
        self.reject('session.redo',code='redo_unavailable')
        self.call('session.restore',{'checkpoint':'baseline'})
        self.assertEqual(self.service.session.active().state.global_iteration,2)
    def test_configuration_rollback(self):
        self.initialize(); self.load(); self.adam(); self.save()
        self.call('session.update',{'updates':{'energy_weight':0.7}})
        self.assertEqual(self.service.session._context(None).configuration,1)
        self.call('session.undo')
        self.assertEqual(self.service.session._context(None).configuration,0)
        self.assertEqual(self.service.session.active().state.global_iteration,2)
    def test_physical_save_and_restore_resets_memory(self):
        self.initialize(); self.load(); self.adam()
        ref=opt.ObjectRef('r',('controls',))
        cp=self.call('session.save',{'name':'physical','mode':'controls_only'}).result
        self.reject('session.restore',{'checkpoint':'physical'},'unsupported_continuation')
        self.call('session.restore',{'checkpoint':cp,'mode':'controls_only'})
        self.adam(1); self.assertEqual(self.service.session.active().state.global_iteration,1)
    def test_uncertified_state_requires_explicit_physical_mode(self):
        self.initialize(); self.load(checkpoint=False); self.adam()
        self.reject('session.save',{'name':'bad'},'unsupported_continuation')
        path=self.service._persistence.store.path; self.call('project.close')
        self.reject('project.open',{'path':str(path)},'unsupported_continuation')
        self.call('project.open',{'path':str(path),'mode':'controls_only'})
        self.assertIsInstance(self.service.session.active(),opt.Controls)
        self.assertIsInstance(self.service.session.get('r'),opt.OptimizerResult)
        self.assertIsNone(self.service.session._objects[self.service.session._bindings['r']].method_key)
        self.reopen(); self.assertIsInstance(self.service.session.active(),opt.Controls)
    def test_callbacks_rejected_without_partial_binding(self):
        self.initialize(); self.load()
        with self.assertRaises(Exception): self.service.register_reference('callback',lambda:None)
        self.assertNotIn('callback',self.service.session.snapshot()['bindings'])
    def test_checkpoint_kind_rejects_incomplete_state(self):
        self.initialize(); self.load(); self.adam()
        self.reject('checkpoints.save',{'value':opt.ObjectRef('r',('controls',)),'kind':'full_state'},'invalid_state')
    def test_readonly_and_writer_lease_contention(self):
        self.initialize(); self.load(); self.adam()
        path=self.service._persistence.store.path
        other=opt.OperationService(); self.services.append(other)
        response=other.execute(opt.OperationRequest('project.open',{'path':str(path)}))
        self.assertFalse(response.ok); self.assertEqual(response.error['code'],'project_locked')
        self.call('project.close')
        self.call('project.open',{'path':str(path),'read_only':True})
        self.call('storage.status')
        self.reject('adam',{'maxiter':1},'read_only')
    def test_storage_full_keeps_previous_head(self):
        self.initialize(); self.load(); self.adam()
        store=self.service._persistence.store
        store.index['policy']['max_bytes']=store.usage()+1024
        self.reject('adam',{'maxiter':2,'step_size':0.01},'storage_full')
    def test_injected_index_write_failure_rolls_back(self):
        self.initialize(); self.load(); self.adam()
        original=store_module._atomic
        def fail_index(path,payload):
            if path.name=='session.json': raise OSError('injected index failure')
            return original(path,payload)
        with patch.object(store_module,'_atomic',side_effect=fail_index):
            self.reject('adam',{'maxiter':2,'step_size':0.01},'persistence_failed')
        self.reopen(); self.assertEqual(self.service.session.active().state.global_iteration,2)
    def test_source_dependency_change_rejected(self):
        dependency=self.root/'input.txt'; dependency.write_text('original')
        self.initialize()
        self.call('session.load',{'name':'system','factory':str(_SOURCE),'parameters':{'N':9},
             'checkpoint':{'stateless':True,'dependencies':[str(dependency)]}})
        self.call('guesses.constant_guess',bind='u',activate=True)
        self.adam(); self.save(); dependency.write_text('changed')
        self.reject('session.restore',{'checkpoint':'baseline'},'source_changed')
    def test_corrupt_blob_and_manifest_rejected(self):
        self.initialize(); self.load(); self.adam(); saved=self.save().result
        store=self.service._persistence.store; identity=saved['checkpoint_id']
        entry=store.index['objects'][identity]
        path=store.path/f'checkpoints/blobs/{entry["blobs"][0]}.npz'
        old=path.read_bytes(); path.write_bytes(b'corrupt')
        self.reject('session.restore',{'checkpoint':'baseline'})
        path.write_bytes(old)
        manifest=store.path/f'checkpoints/manifests/{identity}.json'
        manifest.write_text('{}')
        self.reject('session.restore',{'checkpoint':'baseline'},'checkpoint_corrupt')
    def test_dedup_and_pin_pruning_stale_plan(self):
        self.initialize(); self.load(); self.adam(); saved=self.save().result
        store=self.service._persistence.store
        active=store.index['contexts']['system']['head']
        self.assertEqual(store.index['objects'][active]['blobs'],store.index['objects'][saved['checkpoint_id']]['blobs'])
        self.reject('storage.plan',{'action':'delete','ids':[saved]},'protected_payload')
        plan=self.call('storage.plan',{'action':'prune'},bind='cleanup').object_refs[0]
        self.call('storage.apply',{'plan':plan})
        self.reject('storage.apply',{'plan':plan},'stale_plan')
        self.reject('checkpoints.load',{'checkpoint':'u'},'checkpoint_missing')
    def test_recovery_cleans_prepared_orphans_without_replay(self):
        self.initialize(); self.load(); self.adam()
        path=self.service._persistence.store.path
        self.call('project.close')
        (path/'checkpoints/manifests'/('0'*32+'.json')).write_text('{}')
        (path/'checkpoints/blobs'/('0'*64+'.npz')).write_bytes(b'orphan')
        (path/'.prepare-orphan').write_bytes(b'partial')
        orphan=path/'inputs'/('f'*64); orphan.mkdir(); (orphan/'payload').write_bytes(b'uncommitted')
        self.call('project.open',{'path':str(path)})
        self.assertFalse((path/'.prepare-orphan').exists())
        self.assertFalse(orphan.exists())
        self.assertEqual(self.service.session.active().state.global_iteration,2)
    def test_pruned_names_are_tombstones_in_ram_and_after_open(self):
        self.initialize(); self.load(); self.adam(); self.save()
        plan=self.call('storage.plan',bind='cleanup').object_refs[0]
        self.call('storage.apply',{'plan':plan})
        self.reject('session.get',{'reference':'u'},'payload_pruned')
        self.reopen()
        self.reject('session.get',{'reference':'u'},'payload_pruned')
        self.assertEqual(self.service.session.active().state.global_iteration,2)

    def test_post_replace_sync_failure_keeps_committed_index_and_payloads(self):
        self.initialize(); self.load(); self.adam()
        original=store_module._atomic
        def fail_after_replace(path,payload):
            original(path,payload)
            if path.name=='session.json': raise store_module._CommittedWriteError('injected directory fsync')
        with patch.object(store_module,'_atomic',side_effect=fail_after_replace):
            response=self.adam()
        self.assertIn('fsync',response.metadata['durability_warning'])
        self.assertEqual(self.service.session.active().state.global_iteration,4)
        self.reopen(); self.assertEqual(self.service.session.active().state.global_iteration,4)

    def test_cleanup_plan_shared_blobs_and_report_input_provenance(self):
        self.initialize(); self.load(); self.adam(); self.save()
        self.call('checkpoints.save',{'value':opt.ObjectRef('r'),'kind':'full_state','pin':False})
        report=self.call('utils.verify_gradient',{'directions':1},bind='report')
        saved=self.call('checkpoints.save',{'value':report.object_refs[0],'name':'report-save','kind':'report'}).result
        data=self.service._persistence.store.decode(saved['checkpoint_id'],self.service.session.max_bytes)
        self.assertTrue(data['inputs'])
        plan=self.call('storage.plan').result
        self.assertGreater(plan['reclaimable_bytes'],0)
        self.assertTrue(plan['shared_blobs_kept'])
        self.assertIn('system',self.call('storage.status').result['undo_available'])

    def test_process_termination_before_and_after_index_commit(self):
        self.initialize(); self.load(); self.adam()
        path=self.service._persistence.store.path
        self.call('project.close')
        for point,expected in [('before',2),('after',3)]:
            script=f"""import os
import optimizer as opt
from optimizer.persistence import store
s=opt.OperationService()
assert s.execute(opt.OperationRequest('project.open',{{'path':{str(path)!r}}})).ok
original=store._atomic
def stop(path,payload):
    if path.name=='session.json' and {point!r}=='before': os._exit(71)
    original(path,payload)
    if path.name=='session.json': os._exit(72)
store._atomic=stop
s.execute(opt.OperationRequest('adam',{{'maxiter':1,'step_size':0.01,'gradient_tolerance':None}},bind='r'))
"""
            run=subprocess.run([sys.executable,'-B','-c',script],cwd=_ROOT,capture_output=True,text=True,timeout=20)
            self.assertEqual(run.returncode,71 if point=='before' else 72,run.stderr)
            self.call('project.open',{'path':str(path)})
            self.assertEqual(self.service.session.active().state.global_iteration,expected)
            self.call('project.close')

    def test_noncertified_variant_cannot_claim_exact(self):
        self.initialize(); self.load()
        self.call('adam',{'maxiter':1,'variant':'adamw'},bind='r')
        self.reject('session.save',{'name':'false-exact'},'unsupported_continuation')

    def test_environment_change_and_wrong_physical_family_rejected(self):
        self.initialize(); self.load(); self.adam(); self.save()
        with patch('optimizer.persistence.controller._environment',return_value={'different':True}):
            self.reject('session.restore',{'checkpoint':'baseline'},'environment_changed')
        self.reject('session.restore',{'checkpoint':'baseline','mode':'variables_only'},'invalid_arguments')

    def test_undeclared_local_dependency_rejected(self):
        source=self.root/'system.py'; helper=self.root/'helper.py'
        source.write_text(_SOURCE.read_text().replace('from __future__ import annotations','from __future__ import annotations\nimport helper'))
        helper.write_text('VALUE=1\n')
        sys.path.insert(0,str(self.root))
        try:
            self.initialize()
            self.reject('session.load',{'name':'main','factory':str(source),'checkpoint':{'stateless':True,'dependencies':[]}},'unsupported_persistence')
        finally:
            sys.path.remove(str(self.root)); sys.modules.pop('helper',None)

    def test_scheduled_adam_roundtrip_preserves_position(self):
        self.initialize(); self.load()
        args={'maxiter':2,'step_size':opt.ExponentialDecaySchedule(initial_step=0.02,decay=0.8),'gradient_tolerance':None}
        self.call('adam',args,bind='r'); self.save(); self.reopen()
        self.call('adam',args,bind='r')
        native=opt.adam(self.service.session._context(None).system,self.service.session.get('u'),**dict(args,maxiter=4))
        np.testing.assert_array_equal(self.service.session.active().controls.as_matrix(),native.controls.as_matrix())

    def test_array_factory_parameters_reopen_without_head(self):
        source=self.root/'array_system.py'
        source.write_text(_SOURCE.read_text()+"\ndef array_system(weights):\n    return QuadraticVectorSystem(N=len(weights))\n")
        self.initialize()
        self.call('session.load',{'name':'array','factory':str(source),'factory_name':'array_system',
            'parameters':{'weights':np.arange(9.)},'checkpoint':{'stateless':True,'dependencies':[]}})
        self.reopen()
        self.assertEqual(self.service.session._context(None).layout.control_dim,9)
        self.call('guesses.constant_guess',bind='u',activate=True)

    def test_reentrant_revision_during_preparation_rejects_candidate(self):
        self.initialize(); self.load(); self.adam()
        old_head=self.service.session.snapshot()['contexts']['system']['head']
        store=self.service._persistence.store; generation=store.index['generation']
        original=store_module._atomic; touched=False
        def outside_write(path,payload):
            nonlocal touched
            if not touched and path.parent.name=='manifests':
                touched=True
                self.service.session.select('system')
            return original(path,payload)
        with patch.object(store_module,'_atomic',side_effect=outside_write):
            result=self.service.execute(opt.OperationRequest('adam',{'maxiter':1,'step_size':0.01,'gradient_tolerance':None},bind='r'))
        self.assertFalse(result.ok); self.assertEqual(result.error['code'],'revision_conflict')
        self.assertEqual(generation,store.index['generation'])
        self.assertEqual(old_head,self.service.session.snapshot()['contexts']['system']['head'])

    def test_cli_process_exit_and_reopen(self):
        command=f'opt.project.init(name="cli",root={str(self.root)!r}); opt.session.load(name="main",factory={str(_SOURCE)!r},parameters={{"N":9}},checkpoint={{"stateless":True,"dependencies":[]}}); u=opt.guesses.constant_guess(); opt.session.use(u); r=opt.adam(maxiter=2); saved=opt.session.save(name="baseline"); opt.session.exit();'
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        first=subprocess.run([sys.executable,'-B','-m','optimizer.cli','--command',command,'--json'],cwd=_ROOT,
                              env=env,capture_output=True,text=True,timeout=20)
        self.assertEqual(first.returncode,0,first.stdout+first.stderr)
        command=f'opt.project.open(path={str(self.root/".optimizer/cli")!r}); r=opt.adam(maxiter=2); r.state.global_iteration; opt.session.exit();'
        second=subprocess.run([sys.executable,'-B','-m','optimizer.cli','--command',command,'--json'],cwd=_ROOT,
                               env=env,capture_output=True,text=True,timeout=20)
        self.assertEqual(second.returncode,0,second.stdout+second.stderr)
        records=[json.loads(line) for line in second.stdout.splitlines()]
        self.assertEqual(records[2]['metadata']['summary'],4)
        self.assertTrue((self.root/'.optimizer/cli/history.jsonl').exists())

class CodecTests(unittest.TestCase):
    def test_complex_dtype_zero_dim_and_typed_roundtrip(self):
        value={'scalar':np.array(2.),'array':np.array([1+2j]),'pair':(1,2),'complex':2-3j}
        encoder=_Codec(); encoded=encoder.encode(value)
        decoded=_Codec(encoder.blobs.__getitem__).decode(encoded)
        self.assertEqual(decoded['scalar'].shape,())
        np.testing.assert_array_equal(decoded['array'],value['array'])
        self.assertEqual(decoded['complex'],value['complex'])
    def test_unknown_types_object_array_and_decoded_budget_rejected(self):
        for value in [lambda:None,np.array([object()]),object()]:
            with self.assertRaises(Exception): _Codec().encode(value)
        encoder=_Codec(); encoded=encoder.encode(np.ones(128))
        with self.assertRaises(Exception): _Codec(encoder.blobs.__getitem__,max_bytes=100).decode(encoded)
        with self.assertRaises(Exception): _Codec().decode({'$type':'evil.module.Class','fields':{}})
