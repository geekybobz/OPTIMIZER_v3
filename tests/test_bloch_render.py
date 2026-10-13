"""Bloch contract, camera, export and frontend transaction behavior."""
from pathlib import Path
import hashlib
import tempfile
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optimizer as opt
from optimizer.operations import OperationService, OperationRequest
from optimizer.system_olss import ShootingVariables
from tests.fixtures.shooting_systems import CubicRootShootingSystem
from tests.test_system_spec_visualization import GenericQubitSystem


def trajectory(n=21):
    t=np.linspace(0,1,n)
    return np.column_stack((np.sin(np.pi*t),np.zeros(n),np.cos(np.pi*t))),t


class VisualShooting(CubicRootShootingSystem):
    def plot_data(self,variables,*,kind,**options):
        self.last_options=options
        r,t=trajectory()
        return opt.PlotData(kind=kind,layers=(opt.PlotLayer('shooting',{'points':r,'parameter':t}),))


class BlochTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_raw_coordinates_camera_and_no_mutation(self):
        r,t=trajectory();before=r.copy()
        fig,ax=opt.visualization.plot_bloch(r,parameter=t,projections=False,appearance='paper',azimuth=396,elevation=22,arrows=2,axes=True)
        fig.canvas.draw()
        self.assertEqual(36,ax.azim);self.assertEqual(22,ax.elev)
        np.testing.assert_array_equal(before,r)
        ax.view_init(elev=14,azim=120);fig.canvas.draw()
        np.testing.assert_array_equal(before,r)
        self.assertEqual(120,ax.azim)

    def test_real_finite_radius_validation(self):
        for points in [np.array([['nan','0','0']],dtype=object),np.array([[1j,0,0]]),np.array([[np.inf,0,0]]),np.array([[1.01,0,0]])]:
            with self.subTest(points=points),self.assertRaises(ValueError):
                opt.visualization.plot_bloch(points,projections=False)
        fig,ax=opt.visualization.plot_bloch(np.array([[.1,.2,.3]]),projections=False,appearance='paper')
        self.assertIsNotNone(ax)

    def test_matching_parameter_and_camera_validation(self):
        r,t=trajectory()
        for kwargs in [{'parameter':t[:-1]},{'parameter':t.astype(complex)},{'azimuth':float('nan')},{'elevation':float('inf')},{'arrows':-1},{'color_range':(1,0)}]:
            with self.subTest(kwargs=kwargs),self.assertRaises((ValueError,TypeError)):
                opt.visualization.plot_bloch(r,projections=False,**kwargs)

    def test_shared_time_normalization_across_different_ranges(self):
        r,t=trajectory()
        data=opt.PlotData(kind='bloch',layers=(opt.PlotLayer('a',{'points':r,'parameter':t}),opt.PlotLayer('b',{'points':r*.8,'parameter':2*t})))
        fig,ax=opt.visualization.plot_bloch(data,projections=False,appearance='paper',markers=False)
        colored=[artist for artist in ax.collections if artist.get_array() is not None]
        self.assertEqual(2,len(colored))
        self.assertTrue(all((a.norm.vmin,a.norm.vmax)==(0,2) for a in colored))
        self.assertEqual(2,len(fig.axes))

    def test_incompatible_colorbar_is_rejected(self):
        r,t=trajectory()
        data=opt.PlotData(kind='bloch',layers=(opt.PlotLayer('a',{'points':r,'parameter':t},style={'cmap':'Blues'}),opt.PlotLayer('b',{'points':r,'parameter':t},style={'cmap':'Reds'})))
        with self.assertRaisesRegex(ValueError,'shared colorbar'):
            opt.visualization.plot_bloch(data,appearance='paper')
        fig,ax=opt.visualization.plot_bloch(data,projections=False,appearance='paper',cmap='viridis_r')
        self.assertIsNotNone(ax)

    def test_end_and_target_remain_distinct_and_annotation_tracks_view(self):
        r,t=trajectory()
        data=opt.PlotData(kind='bloch',layers=(opt.PlotLayer('r',{'points':r,'parameter':t}),),annotations={'target':{'point':[0,0,1],'label':'desired target','offset_points':[12,4]}})
        fig,ax=opt.visualization.plot_bloch(data,projections=False,appearance='paper')
        fig.canvas.draw();label=next(text for text in ax.texts if text.get_text()=='desired target')
        before=label.xy
        ax.view_init(elev=40,azim=50);fig.canvas.draw()
        self.assertNotEqual(before,label.xy)
        np.testing.assert_array_equal(data.layers[0].coordinates['points'],r)

    def test_olgs_and_olss_plot_data_use_native_candidates(self):
        gs=GenericQubitSystem();controls=opt.Controls.zeros(gs.control_spec())
        self.assertEqual('bloch',opt.plot_data(gs,controls,kind='bloch').kind)
        ss=VisualShooting();z=ShootingVariables(ss.shooting_spec(),np.zeros(ss.shooting_spec().size))
        data=opt.plot_data(ss,z,kind='bloch',options={'steps':40})
        self.assertEqual('shooting',data.layers[0].name)
        self.assertEqual({'options':{'steps':40}},ss.last_options)
        with self.assertRaises(TypeError):opt.plot_data(ss,controls,kind='bloch')

    def test_artifact_receipt_and_existing_file_preservation(self):
        r,t=trajectory()
        with tempfile.TemporaryDirectory() as d:
            stem=str(Path(d)/'sphere')
            receipt=opt.visualization.render_bloch(r,stem,parameter=t,options={'azimuth':36,'arrows':2},dpi=100)
            self.assertEqual('optimizer.bloch-render/1',receipt['schema'])
            self.assertEqual(0,receipt['numerical_calls'])
            self.assertEqual({'png','pdf'},{a['format'] for a in receipt['artifacts']})
            for a in receipt['artifacts']:
                self.assertEqual(a['sha256'],hashlib.sha256(Path(a['path']).read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):opt.visualization.render_bloch(r,stem,parameter=t)
            self.assertFalse(list(Path(d).glob('.bloch-*')))
            self.assertFalse(plt.get_fignums())

    def test_failed_render_creates_no_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            stem=str(Path(d)/'new'/'sphere')
            with self.assertRaises(ValueError):
                opt.visualization.render_bloch([[2,0,0]],stem)
            self.assertFalse((Path(d)/'new').exists())

    def test_persistent_plot_data_survives_reopen_and_renders_by_reference(self):
        from optimizer.cli.language import _Terminal
        r,t=trajectory()
        data=opt.PlotData(kind='bloch',layers=(opt.PlotLayer('r',{'points':r,'parameter':t}),)).to_dict()
        with tempfile.TemporaryDirectory() as d:
            terminal=_Terminal()
            command=(f"opt.project.init('typed',root={d!r});"
                     f"data=opt.visualization.coerce_plot_data(value={data!r});"
                     "opt.checkpoints.save(value=data,name='trajectory',kind='input',pin=True);")
            self.assertEqual(0,terminal.submit(command))
            terminal.finish();terminal.service.close()
            reopened=_Terminal()
            try:
                path=str(Path(d)/'.optimizer/typed')
                self.assertEqual(0,reopened.submit(f"opt.project.open(path={path!r});"))
                actual=reopened.service.session.get('data')
                self.assertIsInstance(actual,opt.PlotData)
                np.testing.assert_array_equal(r,actual.layers[0].coordinates['points'])
                actual.layers[0].coordinates['points'][0] = 0
                np.testing.assert_array_equal(r,reopened.service.session.get('data').layers[0].coordinates['points'])
                stem=str(Path(d)/'reopened')
                self.assertEqual(0,reopened.submit(f"image=opt.visualization.render_bloch(value=data,output={stem!r},formats=['png'],dpi=100);"))
                self.assertTrue(Path(stem+'.png').exists())
            finally:
                reopened.finish();reopened.service.close()

    def test_native_operation_exports_without_activation(self):
        r,t=trajectory();service=OperationService()
        with tempfile.TemporaryDirectory() as d:
            args={'value':r.tolist(),'parameter':t.tolist(),'output':str(Path(d)/'cli'),'formats':['png'],'dpi':100}
            before=service.session.snapshot()
            result=service.execute(OperationRequest('visualization.render_bloch',args,bind='image'))
            self.assertTrue(result.ok,result.error)
            self.assertTrue(Path(result.result['artifacts'][0]['path']).exists())
            self.assertEqual(before['selected'],service.session.snapshot()['selected'])
            self.assertEqual('artifact',service.info('visualization.render_bloch')['effect'])
            denied=service.execute(OperationRequest('visualization.render_bloch',{**args,'output':str(Path(d)/'bad')},activate=True))
            self.assertFalse(denied.ok);self.assertFalse((Path(d)/'bad.png').exists())

if __name__=='__main__':unittest.main()
