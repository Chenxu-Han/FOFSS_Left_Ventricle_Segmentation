import math
from monai.networks.blocks import Convolution
from monai.networks.nets import FullyConnectedNet

import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal, kl_divergence
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from Code_FOFSS_Echocardiographic_Sequence_Segmentation.torchfde.torchfde.explicit_solver import GrunwaldLetnikov, SolverConfig


class FOFSS(nn.Module):
    def __init__(self, Hidden_Dim=512, Rule_Count=12, History_Length=100, Time_Step=1.0, Alpha_Mean=0.184, Delta_Mean=0.023, Frame_Batch=8):
        super().__init__()
        self.Frame_Batch = Frame_Batch
        self.Image_Encoder = Image_Encoder(Hidden_Dim)
        self.Mask_Encoder = Image_Encoder(Hidden_Dim)
        self.Decoder = Image_Decoder(Hidden_Dim)
        self.State = Fractional_State(Hidden_Dim, Rule_Count, History_Length, Time_Step, Alpha_Mean, Delta_Mean)  # FTKM, Eqs. (3)-(17).
        self.Initial_Prior = nn.Linear(Hidden_Dim, Hidden_Dim)
        self.Initial_Posterior = nn.Linear(2 * Hidden_Dim, 2 * Hidden_Dim)
        self.Posterior = FullyConnectedNet(3 * Hidden_Dim, 2 * Hidden_Dim, [Hidden_Dim], act='RELU')
        self.Time_Head = nn.Linear(Hidden_Dim, 1)
        self.init_weight()

    def init_weight(self):
        for Layer in self.modules():
            if isinstance(Layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(Layer.weight)
                nn.init.zeros_(Layer.bias)
        nn.init.zeros_(self.Posterior.output.weight)
        with torch.no_grad():
            self.Initial_Posterior.weight[self.State.Hidden_Dim:].zero_()
        self.State.Project_Stable()

    def Frame_Forward(self, Module, Frames):
        Parts = []
        for Part in Frames.split(self.Frame_Batch):
            if self.training and torch.is_grad_enabled():
                Parts.append(checkpoint(Module, Part, use_reentrant=False))
            else:
                Parts.append(Module(Part))
        return torch.cat(Parts)

    def forward(self, Images, Masks=None):
        Batch, Length, _, Height, Width = Images.shape
        Control = self.Frame_Forward(self.Image_Encoder, Images.flatten(0, 1)).reshape(Batch, Length, -1)
        Kernel, Inverse, Inputs, Weights, Config, Prior_Scale = self.State.Prepare(Control, Masks is not None)
        if Masks is not None:
            Mask_Features = self.Frame_Forward(self.Mask_Encoder, Masks.flatten(0, 1)).reshape(Batch, Length, -1)
        History, Predicted_History, Divergence = [], [], []
        for Time in range(Length):
            if Time == 0:
                Predicted_Mean = self.Initial_Prior(Control[:, 0])
            else:
                Local = self.State.Transition(Predicted_History, Kernel, Inverse, Inputs[:, Time], Config)
                Predicted_Mean = (Local * Weights[:, Time, :, None]).sum(1)  # Eq. (3), image-only rollout.
            Predicted_History.append(Predicted_Mean)
            if Masks is not None:
                if Time == 0:
                    Prior_Mean = Predicted_Mean
                    Mean, Log_Variance = self.Initial_Posterior(torch.cat([Control[:, 0], Mask_Features[:, 0]], dim=-1)).chunk(2, dim=-1)
                else:
                    Local = self.State.Transition(History, Kernel, Inverse, Inputs[:, Time], Config)
                    Prior_Mean = (Local * Weights[:, Time, :, None]).sum(1)
                    Correction, Log_Variance = self.Posterior(torch.cat([Prior_Mean, Control[:, Time], Mask_Features[:, Time]], dim=-1)).chunk(2, dim=-1)
                    Mean = Prior_Mean + Correction
                Noise_Index = int(Time > 0)
                Std = (0.5 * Log_Variance).exp()
                Posterior = MultivariateNormal(Mean, scale_tril=torch.diag_embed(Std), validate_args=False)
                Prior = MultivariateNormal(Prior_Mean, scale_tril=Prior_Scale[Noise_Index], validate_args=False)
                Divergence.append(kl_divergence(Posterior, Prior))  # Eq. (20), posterior-history Gaussian KL.
                History.append(Normal(Mean, Std, validate_args=False).rsample())
        Output = {'Probability': self.Decode(torch.stack(Predicted_History, dim=1), Control, Height, Width)}
        if Masks is not None:
            States = torch.stack(History, dim=1)
            Output['Control'], Output['Weights'] = Control, Weights
            Output['Posterior_Probability'] = self.Decode(States, Control, Height, Width)
            Output['KL'] = torch.stack(Divergence, dim=1)
            Output['Time'] = self.Time_Head(States).squeeze(-1)
        return Output

    def Decode(self, States, Control, Height, Width):
        Batch, Length = States.shape[:2]
        Features = torch.cat([States, Control], dim=-1).flatten(0, 1)
        Logits = self.Frame_Forward(self.Decoder, Features)
        Logits = F.interpolate(Logits, size=(Height, Width), mode='bilinear', align_corners=False)
        return torch.sigmoid(Logits.reshape(Batch, Length, 1, Height, Width))  # Eq. (18).


class Image_Encoder(nn.Sequential):
    def __init__(self, Hidden_Dim):
        super().__init__(
            Convolution(2, 1, 32, strides=1, act='RELU', norm=None),
            Convolution(2, 32, 64, strides=2, act='RELU', norm=None),
            Convolution(2, 64, 128, strides=2, act='RELU', norm=None),
            Convolution(2, 128, 256, strides=2, act='RELU', norm=None),
            nn.AdaptiveAvgPool2d((8, 8)),
            FullyConnectedNet(256 * 8 * 8, Hidden_Dim, []),
        )


class Image_Decoder(nn.Sequential):
    def __init__(self, Hidden_Dim):
        super().__init__(
            FullyConnectedNet(2 * Hidden_Dim, 256 * 8 * 8, []),
            nn.ReLU(), nn.Unflatten(1, (256, 8, 8)),
            Convolution(2, 256, 128, strides=2, kernel_size=4, padding=1, output_padding=0, is_transposed=True, act='RELU', norm=None),
            Convolution(2, 128, 64, strides=2, kernel_size=4, padding=1, output_padding=0, is_transposed=True, act='RELU', norm=None),
            Convolution(2, 64, 32, strides=2, kernel_size=4, padding=1, output_padding=0, is_transposed=True, act='RELU', norm=None),
            Convolution(2, 32, 1, strides=2, kernel_size=4, padding=1, output_padding=0, is_transposed=True, conv_only=True),
        )


class Fractional_State(nn.Module):
    def __init__(self, Hidden_Dim=512, Rule_Count=12, History_Length=100, Time_Step=1.0, Alpha_Mean=0.184, Delta_Mean=0.023):
        super().__init__()
        self.Hidden_Dim = Hidden_Dim
        self.GL = GrunwaldLetnikov()
        self.History_Length = History_Length
        self.Time_Step = Time_Step
        self.Alpha_Mean = Alpha_Mean
        self.Delta_Mean = Delta_Mean
        self.Alpha_Shift = nn.Parameter(torch.zeros(Rule_Count))
        self.Log_Delta = nn.Parameter(torch.full((Rule_Count,), math.log(Delta_Mean)))
        self.A_Offset = nn.Parameter(torch.randn(Rule_Count, Hidden_Dim, Hidden_Dim) * 0.001)
        self.B_Offset = nn.Parameter(torch.randn(Rule_Count, Hidden_Dim, Hidden_Dim) * 0.001)
        self.Centers = nn.Parameter(torch.randn(Rule_Count, Hidden_Dim) / math.sqrt(Hidden_Dim))
        self.Log_Bandwidth = nn.Parameter(torch.zeros(()))
        self.Noise_Lower = nn.Parameter(torch.zeros(2, Hidden_Dim, Hidden_Dim))
        with torch.no_grad():
            self.Noise_Lower.diagonal(dim1=-2, dim2=-1).fill_(math.log(math.expm1(1.0)))
        self.register_buffer('Identity', torch.eye(Hidden_Dim))

    def Parameters(self):
        Alpha = torch.sigmoid(self.Alpha_Shift + math.log(self.Alpha_Mean / (1 - self.Alpha_Mean)))
        Delta = self.Log_Delta.exp()
        A = -self.Identity / Delta[:, None, None] + self.A_Offset
        B = -self.Identity / Delta[:, None, None].square() + self.B_Offset
        return Alpha, Delta, A, B

    def Project_Stable(self):
        with torch.no_grad():
            Symmetric = (self.A_Offset + self.A_Offset.transpose(-1, -2)) / 2
            Skew = (self.A_Offset - self.A_Offset.transpose(-1, -2)) / 2
            Values, Vectors = torch.linalg.eigh(Symmetric)
            Projected = (Vectors * Values.clamp(max=0).unsqueeze(-2)) @ Vectors.transpose(-1, -2)
            self.A_Offset.copy_(Projected + Skew)  # Eq. (22): sym(A) <= -I/delta.

    def Prepare(self, Control, With_Posterior=True):
        Alpha, _, A, B = self.Parameters()
        Step = self.Time_Step ** Alpha
        Config = SolverConfig(self.History_Length, self.Time_Step, Alpha, Control.device, Control.dtype, h_beta=Step[:, None])
        self.GL.initialize(Config)
        Kernel = self.GL.compute_weights(self.History_Length - 1, 0, Config).flip(-1)  # Eq. (13).
        Matrix = self.Identity - Step[:, None, None] * A
        Inverse = torch.stack([torch.linalg.solve(Item.contiguous(), self.Identity) for Item in Matrix])
        Inputs = torch.einsum('rde,bte->btrd', B, Control)
        Antecedents = Normal(self.Centers, self.Log_Bandwidth.exp(), validate_args=False)
        Weights = Antecedents.log_prob(Control[:, :, None]).sum(-1).softmax(-1)  # Eq. (4).
        Scale = None
        if With_Posterior:
            Diagonal = F.softplus(self.Noise_Lower.diagonal(dim1=-2, dim2=-1)) + 1e-6
            Scale = self.Noise_Lower.tril(-1) + torch.diag_embed(Diagonal)
        return Kernel, Inverse, Inputs, Weights, Config, Scale

    def History_Convolution(self, History, Kernel):
        Length = min(len(History), self.History_Length)
        Shifted = torch.stack(History[-Length:], dim=1) - History[0][:, None]
        Filters = Kernel[:, :Length].flip(-1).repeat(self.Hidden_Dim, 1).unsqueeze(1)
        Memory = F.conv1d(Shifted.transpose(1, 2), Filters, groups=self.Hidden_Dim)  # Eq. (24).
        return Memory.reshape(Shifted.shape[0], self.Hidden_Dim, -1).transpose(1, 2)

    def Transition(self, History, Kernel, Inverse, Input, Config):
        Memory = self.History_Convolution(History, Kernel)
        Initial = History[0][:, None]
        Update = self.GL.compute_update(Initial, Input, Memory - Initial, Config)
        return torch.einsum('rde,bre->brd', Inverse, Update)  # Eqs. (16)-(17).

    def Log_Prior(self, Alpha_Concentration, Delta_Std, A_Std, B_Std):  # MPAM, Eqs. (11)-(12).
        Alpha = torch.sigmoid(self.Alpha_Shift + math.log(self.Alpha_Mean / (1 - self.Alpha_Mean)))
        Delta = self.Log_Delta.exp()
        Alpha_Prior = torch.distributions.Beta(Alpha.new_tensor(Alpha_Concentration * self.Alpha_Mean), Alpha.new_tensor(Alpha_Concentration * (1 - self.Alpha_Mean)))
        Delta_Prior = torch.distributions.LogNormal(Delta.new_tensor(math.log(self.Delta_Mean)), Delta.new_tensor(Delta_Std))
        Zero = self.A_Offset.new_zeros(())
        A_Prior = Normal(Zero, Zero.new_tensor(A_Std), validate_args=False)
        B_Prior = Normal(Zero, Zero.new_tensor(B_Std), validate_args=False)
        return (Alpha_Prior.log_prob(Alpha).sum() + Delta_Prior.log_prob(Delta).sum() + A_Prior.log_prob(self.A_Offset).sum() + B_Prior.log_prob(self.B_Offset).sum())
