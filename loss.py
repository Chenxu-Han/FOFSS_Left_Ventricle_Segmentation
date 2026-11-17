from importlib import import_module

import torch
from torch.nn import functional as F

dice_coeff = import_module("Code_FOFSS_Echocardiographic_Sequence_Segmentation.Pytorch-UNet.utils.dice_score").dice_coeff


def Segmentation_Loss(Probability, Masks, Valid, Dice_Probability=None):
    Cross_Entropy = F.binary_cross_entropy(Probability[Valid], Masks[Valid], reduction='sum')
    Dice_Probability = Probability if Dice_Probability is None else Dice_Probability
    Dice = 1 - dice_coeff(Dice_Probability[:, :, 0][Valid], Masks[:, :, 0][Valid])
    return Cross_Entropy / len(Masks), Dice * Valid.sum() / len(Masks)


def Training_Loss(Output, Masks, Valid, Text_Features, Model, Args, Phase, Phase_Valid):
    Cross_Entropy, Dice = Segmentation_Loss(Output['Posterior_Probability'], Masks, Valid, Output['Probability'])
    KL = Output['KL'][Valid].sum() / len(Masks)
    Physical = -Model.State.Log_Prior(Args.Alpha_Concentration, Args.Delta_Std, Args.A_Std, Args.B_Std)
    Mean_Control = Output['Control'].masked_fill(~Valid[:, :, None], 0).sum(1) / Valid.sum(1, keepdim=True)
    Text = F.mse_loss(Mean_Control, Text_Features, reduction='sum')
    Text = Text / (2 * Args.Text_Variance * len(Masks))  # CTCM, Eq. (21).
    Rule_Weights = Output['Weights'][Valid]
    Rule_Variance = -Rule_Weights.var(dim=-1, unbiased=False).mean()
    Occupancy = F.relu(Args.Min_Occupancy - Rule_Weights.mean(0)).square().sum()
    Centers = F.normalize(Model.State.Centers, dim=-1)
    Rule_Count = len(Centers)
    Separation = torch.triu((Centers @ Centers.T).square(), diagonal=1).sum()
    Separation = 2 * Separation / max(Rule_Count * (Rule_Count - 1), 1)
    Complete_Frames = Valid & Phase_Valid
    Time_Loss = F.mse_loss(Output['Time'][Complete_Frames], Phase[Complete_Frames], reduction='sum')
    Time_Loss = Time_Loss / Complete_Frames.sum().clamp_min(1)
    Auxiliary = Rule_Variance + Args.Occupancy_Weight * Occupancy + Separation + Time_Loss
    Normalizer = Masks.shape[-1] * Masks.shape[-2]
    Loss = (Cross_Entropy + KL + Physical + Args.Text_Weight * Text + Args.Dice_Weight * Dice + Args.Aux_Weight * Auxiliary) / Normalizer  # Eq. (26).
    Parts = {'BCE': Cross_Entropy, 'KL': KL, 'Physical': Physical, 'Text': Text, 'Dice_Loss': Dice, 'Auxiliary': Auxiliary, 'Rule_Variance': Rule_Variance, 'Occupancy': Occupancy, 'Separation': Separation, 'Phase': Time_Loss}
    return Loss, {Name: Value.detach().item() / Normalizer for Name, Value in Parts.items()}
