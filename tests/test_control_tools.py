"""Analytical signal, ownership, cross-layout and real terminal regressions."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import optimizer as opt
from optimizer.cli.language import _Terminal
from optimizer.checkpoints.codec import _Codec
from optimizer.utils import control_tools as tools
from fixtures.quadratic_system import QuadraticVectorSystem
from fixtures.shooting_systems import RectangularShootingSystem

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/quadratic_system.py"


class RecoverableShootingSystem(RectangularShootingSystem):
    def recover_controls(self, variables, *, times):
        spec = opt.ControlSpec(("drive",), len(times), dt=float(times[1]-times[0]))
        return {"spec": spec, "times": times, "values": variables.values[0] * np.ones(spec.shape)}


def grid(n=5, placement="nodes", start=0, stop=1, units="s"):
    return tools.uniform_control_grid(start=start, stop=stop, count=n, placement=placement, units=units)


def controls(g=None, channels=1, complex_values=False, dtype=None):
    g = grid() if g is None else g
    spec = opt.ControlSpec(tuple(f"channel_{i}" for i in range(channels)), g.count,
                           dtype=dtype or (complex if complex_values else float), dt=g.spacing)
    matrix = np.vstack([(i+1) * (1 + np.asarray(g.times)) for i in range(channels)])
    if complex_values: matrix = matrix * (1+2j)
    return opt.Controls(spec, matrix, meta={"control_grid": g, "metrics": {"stale": 999}})


class NumericalControlToolsTests(unittest.TestCase):
    def test_grid_conventions_and_invalid_declarations(self):
        self.assertAlmostEqual(grid(1001).spacing, .001)
        self.assertAlmostEqual(grid(2001).spacing, .0005)
        self.assertAlmostEqual(grid(2000, "left_cells").spacing, .0005)
        self.assertAlmostEqual(grid(2000).spacing, 1/1999)
        for kwargs in ({"count": True}, {"count": 1}, {"placement": "auto"}, {"stop": 0}, {"max_bytes": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, MemoryError)):
                tools.uniform_control_grid(**({"start":0,"stop":1,"count":5,"placement":"nodes"} | kwargs))
        for times in ((0, 1, .5), (0, float("nan"), 1), (0, 0, 1)):
            with self.assertRaises(ValueError): tools.ControlGrid(times, 0, 1, "nodes")

    def test_linear_resampling_large_grids_and_complex_channels(self):
        for channels in (1, 3, 7):
            for count in (2000, 2001):
                with self.subTest(channels=channels, count=count):
                    a, b = grid(1001), grid(count)
                    original = controls(a, channels, complex_values=True)
                    target = opt.ControlSpec(original.keys, count, complex, b.spacing)
                    result = tools.resample_controls(original, source_grid=a, target_grid=b, target=target)
                    expected = controls(b, channels, complex_values=True)
                    np.testing.assert_allclose(result.controls.u, expected.u, atol=1e-14)
                    back = tools.resample_controls(result.controls, source_grid=b, target_grid=a, target=original.spec)
                    np.testing.assert_allclose(back.controls.u, original.u, atol=1e-14)
                    self.assertEqual(result.source_grids, (a,))
                    self.assertNotIn("metrics", result.controls.meta)
                    result.controls.u[:] = 0
                    self.assertTrue(np.all(np.abs(original.u) > 0))

    def test_constant_interpolation_nonuniform_and_downsampling_report(self):
        a = tools.ControlGrid((0, .1, .8, 1), 0, 1, "nodes", "s")
        c = controls(a); c.u[:] = 7
        b = grid(3)
        result = tools.resample_controls(c, source_grid=a, target_grid=b,
                   target=opt.ControlSpec(c.keys, b.count, dt=b.spacing))
        np.testing.assert_array_equal(result.controls.u, 7)
        self.assertTrue(result.facts["downsampled"])
        self.assertFalse(result.facts["anti_alias_filter_applied"])

    def test_cell_centers_and_explicit_extrapolation(self):
        a, b = grid(5, "cell_centers"), grid(10, "cell_centers")
        c = controls(a); target = opt.ControlSpec(c.keys, b.count, dt=b.spacing)
        with self.assertRaises(ValueError): tools.resample_controls(c, source_grid=a, target_grid=b, target=target)
        linear = tools.resample_controls(c, source_grid=a, target_grid=b, target=target, extrapolation="linear")
        np.testing.assert_allclose(linear.controls.u, controls(b).u)
        held = tools.resample_controls(c, source_grid=a, target_grid=b, target=target, extrapolation="hold")
        self.assertEqual(held.facts["extrapolated_samples"], 2)
        self.assertEqual(held.controls.u[0,0], c.u[0,0])

    def test_resampling_rejects_wrong_layout_units_and_lossy_casts(self):
        a, b = grid(5), grid(9); c = controls(a, complex_values=True)
        for target, other in ((opt.ControlSpec(c.keys,9,dt=b.spacing), b),
                              (opt.ControlSpec(c.keys,9,complex,.9), b),
                              (opt.ControlSpec(("wrong",),9,complex,b.spacing), b),
                              (opt.ControlSpec(c.keys,9,complex,b.spacing), grid(9,units="ms")),
                              (opt.ControlSpec(c.keys,9,int,b.spacing), b)):
            with self.assertRaises(ValueError): tools.resample_controls(c, source_grid=a, target_grid=other, target=target)

    def test_affine_scaling_and_selected_channel_normalization(self):
        c = controls(channels=3, complex_values=True)
        scaled = tools.scale_controls(c, factors={"channel_1":2}, offsets={"channel_1":3}, channels=["channel_1"])
        np.testing.assert_array_equal(scaled.controls.u[1], 2*c.u[1]+3)
        np.testing.assert_array_equal(scaled.controls.u[0], c.u[0])
        for norm in ("max_abs", "l2", "rms", "time_l2", "time_rms"):
            with self.subTest(norm=norm):
                result = tools.normalize_controls(c, norm=norm, value=2, scope="channel")
                values = np.abs(result.controls.u)
                measured = values.max(axis=1) if norm == "max_abs" else np.sqrt(
                    (values**2).sum(axis=1) if norm=="l2" else (values**2).mean(axis=1) if norm=="rms" else
                    (values**2 * tools._quadrature(grid())).sum(axis=1))
                np.testing.assert_allclose(measured, 2)
        global_norm = tools.normalize_controls(c, norm="l2", value=3)
        self.assertAlmostEqual(global_norm.controls.norm(),3)

    def test_zero_and_invalid_scaling(self):
        c = controls(); c.u[:] = 0
        with self.assertRaises(ValueError): tools.normalize_controls(c)
        np.testing.assert_array_equal(tools.normalize_controls(c, zero="keep").controls.u,0)
        for kwargs in ({"factors":float("nan")},{"factors":{"wrong":1}},{"factors":1j}):
            with self.assertRaises(ValueError): tools.scale_controls(c, **kwargs)
        with self.assertRaises(ValueError): tools.normalize_controls(c,value=-1)

    def test_filter_known_frequencies_dc_and_complex(self):
        g = grid(128,"left_cells"); c = controls(g,complex_values=True)
        t = np.asarray(g.times)
        c.u[0] = 2 + np.exp(2j*np.pi*3*t) + .5*np.exp(-2j*np.pi*20*t)
        low = tools.filter_controls(c,grid=g,kind="lowpass",cutoff=5,boundary="periodic")
        np.testing.assert_allclose(low.controls.u[0],2+np.exp(2j*np.pi*3*t),atol=1e-14)
        high = tools.filter_controls(c,grid=g,kind="highpass",cutoff=10,boundary="periodic")
        np.testing.assert_allclose(high.controls.u[0],.5*np.exp(-2j*np.pi*20*t),atol=1e-14)
        band = tools.filter_controls(c,grid=g,kind="bandpass",cutoff=[2,5],boundary="periodic")
        np.testing.assert_allclose(band.controls.u[0],np.exp(2j*np.pi*3*t),atol=1e-14)
        stop = tools.filter_controls(c,grid=g,kind="bandstop",cutoff=[2,5],boundary="periodic")
        np.testing.assert_allclose(stop.controls.u[0],2+.5*np.exp(-2j*np.pi*20*t),atol=1e-14)
        self.assertEqual(low.facts["frequency_units"],"cycles/s")
        spec = opt.utils.control_spectrum(c, high_frequency_cutoff=10)
        self.assertEqual(len(spec["frequencies"]),128)
        self.assertGreater(spec["high_frequency_fraction"][c.keys[0]],0)

    def test_filter_and_smoothing_boundary_policies(self):
        for boundary in ("periodic","reflect","hold"):
            c = controls(complex_values=True); c.u[:] = 2+3j
            smooth = tools.smooth_controls(c,sigma=2,boundary=boundary)
            filtered = tools.filter_controls(c,grid=grid(),kind="lowpass",cutoff=1,boundary=boundary)
            np.testing.assert_allclose(smooth.controls.u,c.u)
            np.testing.assert_allclose(filtered.controls.u,c.u)
        nonuniform=tools.ControlGrid((0,.1,.8,1),0,1,"nodes")
        with self.assertRaises(ValueError): tools.filter_controls(controls(nonuniform),grid=nonuniform,kind="lowpass",cutoff=1,boundary="hold")
        with self.assertRaises(ValueError): tools.smooth_controls(controls(nonuniform),sigma=1,boundary="hold")
        with self.assertRaises(ValueError): tools.smooth_controls(controls(),sigma=1,boundary="automatic")
        with self.assertRaises(ValueError): tools.filter_controls(controls(),grid=grid(),kind="lowpass",cutoff=3,boundary="periodic")

    def test_filter_before_downsampling_preserves_known_low_frequency(self):
        a,b=grid(128,"left_cells"),grid(32,"left_cells")
        c=controls(a); t=np.asarray(a.times)
        c.u[0]=np.sin(2*np.pi*3*t)+np.sin(2*np.pi*40*t)
        f=tools.filter_controls(c,grid=a,kind="lowpass",cutoff=10,boundary="periodic")
        r=tools.resample_controls(f.controls,source_grid=a,target_grid=b,target=opt.ControlSpec(c.keys,32,dt=b.spacing))
        np.testing.assert_allclose(r.controls.u[0],np.sin(2*np.pi*3*np.asarray(b.times)),atol=1e-14)

    def test_mix_and_legacy_guess_compatibility_complex_support(self):
        c = controls(complex_values=True)
        mixed = tools.mix_controls([c,c],weights=[2,-1])
        np.testing.assert_array_equal(mixed.controls.u,c.u)
        np.testing.assert_array_equal(opt.guesses.mix_guess([c,c]).u,c.u)
        scaled = opt.guesses.scale_guess(c,amplitude=2,scale="none")
        np.testing.assert_array_equal(scaled.u,2*c.u)
        for dt,dtype in ((.5,complex),(.25,float)):
            wrong = opt.Controls(opt.ControlSpec(c.keys,5,dtype,dt),np.ones(c.shape))
            with self.assertRaises(ValueError): opt.guesses.mix_guess([c,wrong])
        wrong_grid=controls(grid(5,start=1,stop=2),complex_values=True)
        with self.assertRaises(ValueError): tools.mix_controls([c,wrong_grid],weights=[.5,.5])
        with self.assertRaises(ValueError): tools.mix_controls([c],weights=[float("inf")])

    def test_channel_selection_replacement_and_assembly(self):
        c=controls(channels=7)
        selected=tools.select_channels(c,channels=["channel_6","channel_0"])
        np.testing.assert_array_equal(selected.controls.u,c.u[[6,0]])
        replaced=tools.replace_channel(c,channel="channel_0",replacement=c,replacement_channel="channel_6")
        np.testing.assert_array_equal(replaced.controls.u[0],c.u[6])
        target=opt.ControlSpec(("custom_b","custom_a"),5,dt=.25)
        assembled=tools.assemble_channels([c,selected.controls],channels=[(1,"channel_0"),(0,"channel_6")],target=target)
        np.testing.assert_array_equal(assembled.controls.u,c.u[[0,6]])
        with self.assertRaises(ValueError): tools.select_channels(c,channels=["channel_0","channel_0"])
        with self.assertRaises(ValueError): tools.replace_channel(c,channel="channel_0",replacement=controls(grid(5,start=1,stop=2)),replacement_channel="channel_0")

    def test_join_contiguous_nodes_cells_and_reject_gaps(self):
        for placement in ("nodes","left_cells","cell_centers"):
            a,b=grid(5,placement),grid(5,placement,start=1,stop=2)
            first,second=controls(a),controls(b)
            if placement=="nodes":
                with self.assertRaises(ValueError): tools.join_controls_in_time([first,second],grids=[a,b])
                second.u[:,0]=900
            for policy in ("left","right"):
                joined=tools.join_controls_in_time([first,second],grids=[a,b],boundary=policy)
                self.assertEqual(joined.controls.spec.control_dim,9 if placement=="nodes" else 10)
                self.assertEqual(joined.target_grid.stop,2)
                if placement=="nodes": self.assertEqual(joined.controls.u[0,4],2 if policy=="left" else 900)
        c=controls(grid(5,start=2,stop=3))
        with self.assertRaises(ValueError): tools.join_controls_in_time([controls(),c],grids=[grid(),grid(5,start=2,stop=3)])

    def test_bounds_report_is_not_feasibility_certificate(self):
        c=controls(channels=3)
        clipped=tools.apply_control_bounds(c,lower=0,upper=1)
        np.testing.assert_array_equal(clipped.controls.u,1)
        self.assertEqual(clipped.facts["changed_samples"],14)
        self.assertEqual(clipped.facts["feasibility"],"not_evaluated")
        self.assertGreater(clipped.facts["correction_norm"],0)
        with self.assertRaises(ValueError): tools.apply_control_bounds(c,lower=2,upper=1)
        with self.assertRaises(ValueError): tools.apply_control_bounds(controls(complex_values=True),lower=0,upper=1)

    def test_recovery_hook_and_missing_capability(self):
        system=RecoverableShootingSystem(); variables=opt.ShootingVariables(system.shooting_spec(),[2,0])
        recovered=tools.recover_shooting_controls(system,variables,grid=grid(9))
        np.testing.assert_array_equal(recovered.controls.u,2)
        self.assertFalse(recovered.facts["inverse_mapping"])
        with self.assertRaises(NotImplementedError): tools.recover_shooting_controls(RectangularShootingSystem(),variables,grid=grid())
        with patch.object(system,"recover_controls",return_value={"values":np.ones((1,5))}):
            with self.assertRaises(ValueError): tools.recover_shooting_controls(system,variables,grid=grid())

    def test_preflight_rejects_before_large_allocations(self):
        c=controls()
        with patch.object(np,"convolve",side_effect=AssertionError("computed")):
            with self.assertRaises(MemoryError): tools.smooth_controls(c,sigma=100,boundary="hold",max_bytes=100)
        with patch.object(np.fft,"fft",side_effect=AssertionError("computed")):
            with self.assertRaises(MemoryError): tools.filter_controls(c,grid=grid(),kind="lowpass",cutoff=1,boundary="periodic",max_bytes=100)

    def test_typed_codec_roundtrip(self):
        original=tools.scale_controls(controls(complex_values=True),factors=2)
        codec=_Codec(); encoded=codec.encode(original)
        decoded=_Codec(read_blob=codec.blobs.__getitem__).decode(encoded)
        self.assertIsInstance(decoded,tools.ControlTransformResult)
        self.assertEqual(decoded.target_grid,original.target_grid)
        np.testing.assert_array_equal(decoded.controls.u,original.controls.u)

    def test_float32_precision_and_independent_metadata(self):
        a,b=grid(),grid(9)
        c=controls(a,dtype=np.float32)
        target=opt.ControlSpec(c.keys,9,np.float32,b.spacing,meta={"nested":{"original":True}})
        result=tools.resample_controls(c,source_grid=a,target_grid=b,target=target)
        self.assertEqual(result.controls.u.dtype,np.dtype("float32"))
        np.testing.assert_allclose(result.controls.u,controls(b).u)
        result.controls.spec.meta["nested"]["original"]=False
        self.assertTrue(target.meta["nested"]["original"])
        factors=np.array([2.0]); scaled=tools.scale_controls(c,factors=factors)
        factors[:]=900
        self.assertEqual(scaled.parameters["factors"][0,0],2)
        self.assertEqual(scaled.controls.u.dtype,np.dtype("float32"))

    def test_grid_units_quadrature_and_failed_repair_stays_explicit(self):
        g=grid(5,"cell_centers",stop=2)
        c=controls(g); c.u[:]=4
        normalized=tools.normalize_controls(c,norm="time_l2",value=3)
        self.assertAlmostEqual(float(np.sqrt(np.sum(np.abs(normalized.controls.u)**2*tools._quadrature(g)))),3)
        altered=opt.Controls(opt.ControlSpec(c.keys,5,dt=g.spacing,meta={"units":"A"}),c.u)
        with self.assertRaises(ValueError): tools.mix_controls([c,altered],weights=[1,1],grids=[g,g])
        system=QuadraticVectorSystem(N=5)
        bad=opt.constant_guess(system,value=100)
        clipped=tools.apply_control_bounds(bad,lower=0,upper=0)
        self.assertGreater(opt.utils.metric_report(system,clipped.controls)["metrics"]["residual_norm2"],0)
        repair=opt.utils.repair_newton(system,clipped.controls,maxiter=0,tolerance=1e-14)
        self.assertFalse(repair.converged)
        self.assertEqual(clipped.facts["feasibility"],"not_evaluated")

    def test_unsafe_target_cast_rejected_before_interpolation(self):
        c=controls(complex_values=True)
        with patch.object(np,"interp",side_effect=AssertionError("computed")):
            with self.assertRaises(ValueError): tools.resample_controls(c,source_grid=grid(),target_grid=grid(9),
                target=opt.ControlSpec(c.keys,9,float,.125))


class ControlWorkflowTests(unittest.TestCase):
    def call(self, service, operation, arguments=None, **options):
        response=service.execute(opt.OperationRequest(operation, arguments or {}, **options))
        self.assertTrue(response.ok,response.error)
        return response

    def test_service_cross_grid_validation_activation_and_reset(self):
        service=opt.OperationService()
        self.call(service,"session.load",{"name":"coarse","factory":QuadraticVectorSystem,"parameters":{"N":5}})
        u=self.call(service,"guesses.constant_guess",{"value":.1},bind="u",activate=True)
        self.call(service,"adam",{"maxiter":2})
        coarse_head=service.session.snapshot()["contexts"]["coarse"]["head"]
        self.call(service,"session.load",{"name":"fine","factory":QuadraticVectorSystem,"parameters":{"N":9}})
        target=self.call(service,"control_spec",context="fine").result
        r=self.call(service,"utils.resample_controls",{"controls":u.object_refs[0],"source_grid":grid(),"target_grid":grid(9)},context="fine",bind="converted")
        np.testing.assert_array_equal(r.result.controls.u,.1)
        self.assertIn(u.object_refs[0],r.result.source_refs)
        self.assertIsNone(service.session.snapshot()["contexts"]["fine"]["head"])
        self.assertEqual(service.session.snapshot()["contexts"]["coarse"]["head"],coarse_head)
        before=service.session.snapshot()
        wrong=service.execute(opt.OperationRequest("utils.resample_controls",{"controls":u.result,"source_grid":grid(),"target_grid":grid(),"target":u.result.spec},context="fine"))
        self.assertFalse(wrong.ok); self.assertEqual(wrong.error["code"],"incompatible_layout")
        self.assertEqual(before,service.session.snapshot())
        self.call(service,"session.use",{"value":r.object_refs[0],"context":"fine"})
        adam=self.call(service,"adam",{"maxiter":1},context="fine")
        self.assertEqual(adam.result.state.global_iteration,1)
        scaled=self.call(service,"utils.scale_controls",{"controls":opt.ObjectRef(adam.object_refs[0].object_id,("controls",)),"factors":2})
        self.call(service,"session.use",{"value":scaled.object_refs[0],"context":"fine"})
        reset=self.call(service,"adam",{"maxiter":1},context="fine")
        self.assertEqual(reset.result.state.global_iteration,1)
        denied=service.execute(opt.OperationRequest("utils.scale_controls",{"controls":r.result.controls,"factors":2},activate=True))
        self.assertFalse(denied.ok)

    def test_service_recovery_keeps_shooting_state_and_outputs_controls(self):
        service=opt.OperationService()
        self.call(service,"session.load",{"name":"shot","factory":RecoverableShootingSystem})
        variables=opt.ShootingVariables(RecoverableShootingSystem().shooting_spec(),[2,0])
        self.call(service,"session.use",{"value":variables})
        head=service.session.snapshot()["contexts"]["shot"]["head"]
        r=self.call(service,"utils.recover_shooting_controls",{"grid":grid(9)})
        np.testing.assert_array_equal(r.result.controls.u,2)
        self.assertEqual(service.session.snapshot()["contexts"]["shot"]["head"],head)

    def test_terminal_checkpoint_reopen_replay_and_python_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            emitted=[]; terminal=_Terminal(emit=emitted.append)
            self.addCleanup(terminal.service._persistence.close)
            def run(source):
                self.assertEqual(terminal.submit(source),0,emitted[-1].error if emitted else source)
            run(f"opt.project.init('control-demo',root={directory!r});")
            run(f"opt.session.load('coarse',{str(FIXTURE)!r},parameters={{'N':5}},checkpoint={{'stateless':True,'dependencies':[]}});")
            run("u=opt.guesses.constant_guess(value=.2); opt.session.use(u);")
            run("g=opt.utils.uniform_control_grid(start=0,stop=1,count=5,placement='nodes',units='s');")
            run("f=opt.utils.uniform_control_grid(start=0,stop=1,count=9,placement='nodes',units='s');")
            run(f"opt.session.load('fine',{str(FIXTURE)!r},parameters={{'N':9}},checkpoint={{'stateless':True,'dependencies':[]}});")
            run("target=opt.control_spec(context='fine');")
            run("c=opt.utils.resample_controls(u,source_grid=g,target_grid=f,target=target,context='fine');")
            run("s=opt.utils.smooth_controls(c.controls,sigma=1,boundary='hold');")
            run("n=opt.utils.normalize_controls(s.controls,norm='max_abs',value=.3);")
            run("n.facts; check=opt.utils.metric_report(controls=n.controls,context='fine');")
            native=tools.normalize_controls(tools.smooth_controls(tools.resample_controls(
                opt.constant_guess(QuadraticVectorSystem(N=5),value=.2),source_grid=grid(),target_grid=grid(9),
                target=QuadraticVectorSystem(N=9).control_spec()).controls,sigma=1,boundary="hold").controls,norm="max_abs",value=.3)
            np.testing.assert_array_equal(terminal.service.session.get("n").controls.u,native.controls.u)
            run("saved=opt.checkpoints.save(n,name='candidate',kind='report');")
            run("opt.session.use(n.controls,context='fine'); r=opt.adam(maxiter=1,activate=False,context='fine');")
            run("plan=opt.history.replay_plan(mode='commands'); opt.history.replay(plan=plan,project='replayed');")
            replayed=Path(directory)/'.optimizer'/'replayed'
            self.assertTrue(replayed.exists())
            run("opt.project.close();")
            run(f"opt.project.open({str(Path(directory)/'.optimizer'/'control-demo')!r});")
            np.testing.assert_array_equal(terminal.service.session.get("n").controls.u,native.controls.u)
            self.assertEqual(terminal.service.session.get("n").target_grid,grid(9))
            from optimizer.history.journal import _Journal
            raw=[row['data']['raw'] for row in _Journal(terminal.service._persistence.store).rows() if row['kind']=='submitted']
            self.assertTrue(any("resample_controls" in line for line in raw))
            run("opt.project.close();")
            run(f"opt.project.open({str(replayed)!r});")
            np.testing.assert_array_equal(terminal.service.session.get("n").controls.u,native.controls.u)

    def test_real_cli_process(self):
        source=(f"opt.session.load('main',{str(FIXTURE)!r},parameters={{'N':5}});"
                "u=opt.guesses.constant_guess(value=.1);"
                "g=opt.utils.uniform_control_grid(start=0,stop=1,count=5,placement='nodes');"
                "s=opt.utils.scale_controls(u,factors=2);"
                "f=opt.utils.filter_controls(s.controls,grid=g,kind='lowpass',cutoff=1,boundary='hold');"
                "opt.session.use(f.controls);r=opt.adam(maxiter=1);r.metrics;")
        result=subprocess.run([sys.executable,"-B","-m","optimizer.cli","--json","--command",source],cwd=ROOT,
                              capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        rows=[json.loads(line) for line in result.stdout.splitlines()]
        self.assertIn("J",rows[-1]["metadata"]["summary"])
        self.assertTrue(any(row.get("metadata",{}).get("summary",{}).get("type")=="ControlTransformResult" for row in rows))


if __name__ == "__main__": unittest.main()
