import inspect
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

Root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Root.parent))
from Code_FOFSS_Echocardiographic_Sequence_Segmentation import model as Model_Code
from Code_FOFSS_Echocardiographic_Sequence_Segmentation import loss as Loss_Code
from Code_FOFSS_Echocardiographic_Sequence_Segmentation import dataset as Data_Code
from Code_FOFSS_Echocardiographic_Sequence_Segmentation import metrics as Metrics
from Code_FOFSS_Echocardiographic_Sequence_Segmentation import train as Train_Code

torch.set_num_threads(2)


class FOFSS_Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.Model = Model_Code.FOFSS(4, 3, 3, Time_Step=0.4, Frame_Batch=2)
        self.Images = torch.rand(2, 4, 1, 21, 25)
        self.Masks = (torch.rand_like(self.Images) > 0.7).float()
        self.Valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        self.Args = Train_Code.Parser.parse_args([])
        self.Args.Min_Occupancy = 0.5 / 3
        self.Args.Dice_Weight = self.Args.Image_Size ** 2
        self.Phase = torch.tensor([[0., 1/3, 2/3, 1.], [0., 0., 0., 0.]])
        self.Phase_Valid = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)

    def test_matrix_projection_and_optimizer(self):
        core = self.Model.State.double()
        with torch.no_grad():
            core.A_Offset.copy_(torch.randn_like(core.A_Offset))
            core.Log_Delta.add_(0.7)
        _, delta, a, _ = core.Parameters()
        original = a.detach().numpy()
        expected = []
        for index, matrix in enumerate(original):
            values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
            expected.append((vectors * np.minimum(values, -1 / delta[index].item())) @ vectors.T + (matrix - matrix.T) / 2)
        core.Project_Stable()
        torch.testing.assert_close(core.Parameters()[2], torch.from_numpy(np.array(expected)), rtol=1e-11, atol=1e-11)
        optimizer = torch.optim.Adam(core.parameters(), lr=0.01)
        for step in range(3):
            optimizer.zero_grad()
            objective = -core.A_Offset.diagonal(dim1=-2, dim2=-1).sum() + core.Log_Delta.sum()
            objective.backward()
            optimizer.step()
            core.Project_Stable()
            a = core.Parameters()[2]
            delta = core.Parameters()[1]
            residual = (a + a.transpose(-1, -2)) / 2 + core.Identity / delta[:, None, None]
            self.assertLess(torch.linalg.eigvalsh(residual).max().item(), 1e-10)

    def test_independent_prior_rollout_and_dice_gradients(self):
        model = self.Model
        first = model(self.Images, self.Masks)
        changed = model(self.Images, 1 - self.Masks)
        inference = model(self.Images)
        torch.testing.assert_close(first['Probability'], changed['Probability'], rtol=0, atol=0)
        torch.testing.assert_close(first['Probability'], inference['Probability'], rtol=0, atol=0)
        self.assertFalse(torch.equal(first['Posterior_Probability'], changed['Posterior_Probability']))
        _, dice = Loss_Code.Segmentation_Loss(first['Probability'], self.Masks, self.Valid)
        dice.backward()
        for layer in (model.Initial_Prior, model.Image_Encoder, model.Decoder):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters()))
        for layer in (model.Mask_Encoder, model.Initial_Posterior, model.Posterior):
            self.assertTrue(all(p.grad is None for p in layer.parameters()))

    def test_causal_and_padding_independence(self):
        model = self.Model.eval()
        with torch.no_grad():
            baseline = model(self.Images)['Probability']
            modified = self.Images.clone()
            modified[:, 2:] = torch.rand_like(modified[:, 2:])
            actual = model(modified)['Probability']
            torch.testing.assert_close(actual[:, :2], baseline[:, :2], rtol=0, atol=0)
            short = model(self.Images[1:2, :2])['Probability']
            torch.testing.assert_close(short[0], baseline[1, :2], rtol=1e-5, atol=1e-7)

    def test_loss_terms_and_padding(self):
        output = self.Model(self.Images, self.Masks)
        output['Control'] = torch.tensor([[[1., 0, 0, 0], [-1., 0, 0, 0], [2., 0, 0, 0], [-2., 0, 0, 0]],
                                          [[3., 0, 0, 0], [-3., 0, 0, 0], [float('nan')]*4, [float('nan')]*4]])
        output['Weights'] = torch.tensor([[[1.,0,0]]*4, [[1.,0,0]]*4])
        total, parts = Loss_Code.Training_Loss(output, self.Masks, self.Valid, torch.zeros(2,4), self.Model, self.Args, self.Phase, self.Phase_Valid)
        area = 21 * 25
        self.assertEqual(parts['Text'], 0)
        self.assertAlmostEqual(parts['Rule_Variance'] * area, -2/9, places=6)
        self.assertAlmostEqual(parts['Occupancy'] * area, 2*(1/6)**2, places=6)
        expected_phase = ((output['Time'][0] - self.Phase[0])**2).mean()
        self.assertAlmostEqual(parts['Phase'] * area, expected_phase.item(), places=6)
        expected_bce = torch.nn.functional.binary_cross_entropy(output['Posterior_Probability'][self.Valid], self.Masks[self.Valid], reduction='sum') / 2
        self.assertAlmostEqual(parts['BCE'] * area, expected_bce.item(), places=3)
        total.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.Model.parameters()))

    def test_no_complete_cycles_has_zero_phase_loss(self):
        output = self.Model(self.Images, self.Masks)
        total, parts = Loss_Code.Training_Loss(output, self.Masks, self.Valid, torch.zeros(2,4), self.Model, self.Args,
                                              torch.full((2,4), float('nan')), torch.zeros(2,4,dtype=torch.bool))
        self.assertEqual(parts['Phase'], 0)
        self.assertTrue(torch.isfinite(total))

    def test_gl_exact_implicit_recurrence(self):
        core = self.Model.State.double()
        control = torch.randn(2, 7, 4, dtype=torch.double)
        kernel, inverse, inputs, weights, config, scale = core.Prepare(control)
        alpha, delta, a, b = core.Parameters()
        history = [torch.randn(2,4,dtype=torch.double) for _ in range(6)]
        actual = core.Transition(history, kernel, inverse, inputs[:,6], config)
        for sample in range(2):
            for rule in range(3):
                coefficient, memory = 1., torch.zeros(4,dtype=torch.double)
                for lag in range(1,4):
                    coefficient = coefficient * (1-(1+alpha[rule])/lag)
                    torch.testing.assert_close(kernel[rule,lag-1], coefficient)
                    memory += coefficient*(history[-lag][sample]-history[0][sample])
                step=core.Time_Step**alpha[rule]
                expected=history[0][sample]+torch.linalg.solve(core.Identity-step*a[rule],-memory+step*(b[rule]@control[sample,6]+a[rule]@history[0][sample]))
                torch.testing.assert_close(actual[sample,rule],expected,rtol=1e-10,atol=1e-10)
        self.assertTrue((torch.autograd.grad(kernel.sum(),core.Alpha_Shift)[0].abs()>0).all())

    def test_cycle_mapping_and_corrupt_data(self):
        with tempfile.TemporaryDirectory() as folder:
            sequence=Path(folder)/'clip'
            sequence.mkdir()
            area=[4,16,16,9,4,9,16,16,9,4,9,16,16,4]
            for frame,count in enumerate(area):
                Image.fromarray(np.ones((8,8),dtype=np.uint8)*255).save(sequence/f'{frame:04d}_img.png')
                mask=np.zeros((8,8),dtype=np.uint8)
                mask.flat[:count]=255
                Image.fromarray(mask).save(sequence/f'{frame:04d}_mask_LV.png')
            dataset=Data_Code.Sequence_Dataset([sequence],8,Require_Masks=True)
            item=dataset[0]
            torch.testing.assert_close(item['Phase'],torch.tensor([0.,0.,0.,.2,.4,.6,.8,0.,.2,.4,.6,.8,1.,0.]))
            self.assertEqual(item['Phase_Valid'].tolist(),[False]*2+[True]*11+[False])
            batch=Data_Code.Sequence_Batch([item,{'Path':'bad','Error':'invalid image'}])
            self.assertEqual(len(batch['Skipped']),1)
            self.assertEqual(batch['Phase_Valid'].sum(),11)
            (sequence/'0002_img.png').write_bytes(b'invalid image')
            bad=dataset[0]
            self.assertIn('Error',bad)
            self.assertIsNone(Data_Code.Sequence_Batch([bad])['Images'])

    def test_incomplete_flat_gapped_and_empty_cycles(self):
        for area in ([1,2,3,4,3,2,1], [4,3,2,1,2,3,4], [5]*12, [0]*12, []):
            phase,valid=Data_Code.Cycle_Phase(area,np.arange(len(area)))
            self.assertFalse(valid.any())
            self.assertTrue(torch.isfinite(phase).all())
        area=[4,16,9,4,9,16,4]
        phase,valid=Data_Code.Cycle_Phase(area,[0,1,2,3,5,6,7])
        self.assertFalse(valid.any())
        phase,valid=Data_Code.Cycle_Phase([4,16,9,0,9,16,4],np.arange(7))
        self.assertFalse(valid.any())
        phase,valid=Data_Code.Cycle_Phase([],np.arange(7))
        self.assertFalse(valid.any())
        area=[4,16,12,13,8,4,8,16,4]
        phase,valid=Data_Code.Cycle_Phase(area,np.arange(100,109))
        torch.testing.assert_close(phase[1:8],torch.linspace(0,1,7))
        self.assertEqual(valid.sum(),7)

    def test_patient_split_uses_all_paths(self):
        paths=[Path(f'/Data/Data_Png_All/source/heart/patient_{p}/view_{v}/filename/clip') for p in range(20) for v in range(2)]
        train,val=Data_Code.Split_Training(paths,.2,42)
        self.assertEqual(len(train),32)
        self.assertEqual(len(val),8)
        self.assertEqual(set(train)|set(val),set(paths))
        self.assertFalse({Data_Code.Patient_Id(p) for p in train}&{Data_Code.Patient_Id(p) for p in val})
        self.assertEqual((train,val),Data_Code.Split_Training(paths,.2,42))

    def test_original_resolution_dice(self):
        probability = torch.tensor([[[[[0.2, 0.8], [0.2, 0.8]]], [[[0., 0.], [0., 0.]]], [[[1., 1.], [1., 1.]]]]])
        target = np.zeros((6, 8), dtype=bool)
        target[:, 6:] = True
        score, count = Metrics.Dice_Score(probability, [[target, np.zeros((3, 5), dtype=bool)]], torch.tensor([[True, True, False]]))
        self.assertEqual(count, 2)
        self.assertAlmostEqual(score, 2 / 3 + 1)
        score, count = Metrics.Dice_Score(torch.zeros(1, 1, 1, 2, 2), [[np.ones((9, 7), dtype=bool)]], torch.ones(1, 1, dtype=torch.bool))
        self.assertEqual((score, count), (0, 1))

    def test_empty_prediction_and_nl_upstream(self):
        prediction=np.zeros((5,5),dtype=bool)
        target=np.ones_like(prediction)
        scores=Metrics.Frame_Metrics(prediction,target)
        for name in ('Dice','IoU','Recall'):
            self.assertEqual(scores[name],0)
        self.assertEqual(scores['PAD'],100)
        for name in ('Precision','HD','ASSD'):
            self.assertTrue(math.isnan(scores[name]))
        masks=np.zeros((5,4,4),dtype=bool)
        area=np.array([1,3,8,3,1])
        for i,count in enumerate(area):masks[i].flat[:count]=True
        actual=Metrics.Temporal_Metrics(masks)
        expected=np.r_[0,np.abs(np.diff(area/16,n=2)),0]/0.016
        np.testing.assert_allclose(actual,expected)
        self.assertEqual(Path(inspect.getfile(Metrics.compute_temporal_consistency_metric)).resolve(),Root/'vital/vital/metrics/evaluate/attribute.py')
        self.assertTrue(np.isnan(Metrics.Temporal_Metrics(masks[:2])).all())
        np.testing.assert_allclose(Metrics.Temporal_Metrics(masks.repeat(3,axis=1).repeat(3,axis=2)),actual)
        np.testing.assert_equal(Metrics.Temporal_Metrics(np.zeros_like(masks)),0)
        np.testing.assert_equal(Metrics.Temporal_Metrics(np.ones_like(masks)),0)
        gapped=Metrics.Temporal_Metrics(masks,[0,1,2,5,6])
        np.testing.assert_allclose(gapped[:3],Metrics.Temporal_Metrics(masks[:3]))
        self.assertTrue(np.isnan(gapped[3:]).all())

    def test_reused_core_executes(self):
        from monai.networks.blocks import Convolution
        from monai.networks.nets import FullyConnectedNet
        used=set()
        blocks={n:m for n,m in self.Model.named_modules() if isinstance(m,(Convolution,FullyConnectedNet))}
        handles=[m.register_forward_hook(lambda m,a,o,n=n:used.add(n)) for n,m in blocks.items()]
        with patch.object(self.Model.State.GL,'compute_update',wraps=self.Model.State.GL.compute_update) as gl:
            self.Model(self.Images,self.Masks)
            self.assertEqual(gl.call_count,6)
        self.assertEqual(used,set(blocks))
        self.assertEqual(Train_Code.Engine.__module__, 'ignite.engine.engine')
        for handle in handles:handle.remove()


if __name__ == '__main__':
    unittest.main(verbosity=2)
