import numpy as np
from PIL import Image
from Code_FOFSS_Echocardiographic_Sequence_Segmentation.vital.vital.metrics.evaluate.attribute import compute_temporal_consistency_metric
from medpy.metric.binary import dc, jc, precision, recall, ravd, hd, assd


def Dice_Score(Probability, Original_Masks, Valid):
    Probability = Probability.detach().float().cpu().numpy()
    Score, Count = 0.0, 0
    for Sample, Frame in Valid.nonzero().tolist():
        Target = Original_Masks[Sample][Frame]
        Height, Width = Target.shape
        Restored = np.asarray(Image.fromarray(Probability[Sample, Frame, 0]).resize((Width, Height), Image.Resampling.BILINEAR))
        Prediction = Restored >= 0.5
        Score += dc(Prediction, Target) if Prediction.any() or Target.any() else 1.0
        Count += 1
    return Score, Count


def Frame_Metrics(Prediction, Target):
    Prediction, Target = Prediction.astype(bool), Target.astype(bool)
    Pred_Area, True_Area = Prediction.sum(), Target.sum()
    if Pred_Area == 0 and True_Area == 0:
        Dice, IoU, Precision, Recall, PAD, HD, ASSD = 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0
    elif Pred_Area == 0:
        Dice, IoU, Recall, PAD = 0.0, 0.0, 0.0, 100.0
        Precision, HD, ASSD = float('nan'), float('nan'), float('nan')
    elif True_Area == 0:
        Dice, IoU, Precision = 0.0, 0.0, 0.0
        Recall, PAD, HD, ASSD = float('nan'), float('nan'), float('nan'), float('nan')
    else:
        Dice, IoU = dc(Prediction, Target), jc(Prediction, Target)
        Precision, Recall = precision(Prediction, Target), recall(Prediction, Target)
        PAD = 100 * abs(ravd(Prediction, Target))
        HD, ASSD = hd(Prediction, Target), assd(Prediction, Target)
    return {'Dice': float(Dice), 'IoU': float(IoU), 'Precision': float(Precision), 'Recall': float(Recall), 'PAD': float(PAD), 'HD': float(HD), 'ASSD': float(ASSD)}


NL_Threshold = 0.016


def Temporal_Metrics(Prediction, Frame_Numbers=None):
    Scores = np.full(len(Prediction), np.nan)
    if len(Prediction) < 3:
        return Scores
    Frame_Numbers = np.arange(len(Prediction)) if Frame_Numbers is None else np.asarray(Frame_Numbers)
    Breaks = np.flatnonzero(np.diff(Frame_Numbers) != 1) + 1
    Area = np.array([Mask.sum() / Mask.size for Mask in Prediction], dtype=np.float64)
    for Segment in np.split(np.arange(len(Prediction)), Breaks):
        if len(Segment) < 3:
            continue
        Difference = compute_temporal_consistency_metric(Area[Segment], bounds=(0, 1))
        Scores[Segment] = 2 * np.abs(Difference) / NL_Threshold
    return Scores
