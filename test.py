import argparse
import csv
import json
import sys
from pathlib import Path

Release_Root = Path(__file__).resolve().parent
sys.path.insert(0, str(Release_Root.parent))

from Code_FOFSS_Echocardiographic_Sequence_Segmentation.dataset import Read_Paths, Sequence_Dataset, Sequence_Batch, Validate_Sequences, Record_Skips
from Code_FOFSS_Echocardiographic_Sequence_Segmentation.metrics import Frame_Metrics, Temporal_Metrics
from Code_FOFSS_Echocardiographic_Sequence_Segmentation.model import FOFSS

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

Parser = argparse.ArgumentParser(description='Evaluate FOFSS at the original image resolution.')
Parser.add_argument('--Data_Root', type=Path, default=None, help='Absolute path to the data directory.')
Parser.add_argument('--Test_File', type=Path, default=None, help='Sequence list relative to Data_Root, or an absolute path.')
Parser.add_argument('--Checkpoint', type=Path, default=Path('weights/Weight_FOFSS.pth'))
Parser.add_argument('--Output_Dir', type=Path, default=Path('evaluation'))
Parser.add_argument('--Device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
Parser.add_argument('--Workers', type=int, default=4)


def Test(Args):
    if Args.Data_Root is None or not Args.Data_Root.is_absolute():
        Parser.error('Set --Data_Root to the absolute path of your data directory.')
    Args.Data_Root = Args.Data_Root.resolve()
    Args.Test_File = (Args.Data_Root / Args.Test_File).resolve() if Args.Test_File else Args.Data_Root / 'Data_Split_txt/cross_validation_folds/dir_format/fold_1/test.txt'
    Args.Output_Dir = (Release_Root / Args.Output_Dir).resolve()
    if Args.Output_Dir == Args.Data_Root or Args.Data_Root in Args.Output_Dir.parents:
        Parser.error('Output_Dir must be outside Data_Root.')
    State = torch.load(Release_Root / Args.Checkpoint, map_location='cpu', weights_only=True)
    Model = FOFSS(**State['Model_Config']).to(Args.Device)
    Model.load_state_dict(State['Model'], strict=True)
    Model.eval()
    Args.Output_Dir.mkdir(parents=True, exist_ok=True)
    Skip_File = Args.Output_Dir / 'skipped.csv'
    Paths = Read_Paths(Args.Test_File, Args.Data_Root)
    Paths = Validate_Sequences(Paths, Require_Masks=True, Skip_File=Skip_File)
    Data = Sequence_Dataset(Paths, State['Image_Size'], Require_Masks=False)
    Loader = DataLoader(Data, batch_size=1, collate_fn=Sequence_Batch, num_workers=Args.Workers)
    Names = ['Dice', 'IoU', 'Precision', 'Recall', 'PAD', 'HD', 'ASSD', 'NL']
    Totals, Counts = dict.fromkeys(Names, 0.0), dict.fromkeys(Names, 0)
    with torch.no_grad(), tqdm(Loader, desc='Test') as Progress, (Args.Output_Dir / 'metrics.csv').open('w', encoding='utf-8', newline='') as File:
        Writer = csv.DictWriter(File, fieldnames=['Sequence', 'Frame'] + Names)
        Writer.writeheader()
        for Batch in Progress:
            Record_Skips(Batch['Skipped'], Skip_File)
            if Batch['Images'] is None:
                continue
            Probability = Model(Batch['Images'].to(Args.Device))['Probability'][0, :, 0].float().cpu().numpy()
            Sequence = Path(Batch['Paths'][0])
            Predictions, Rows, Frame_Numbers = [], [], []
            for Index, Name in enumerate(Batch['Names'][0]):
                with Image.open(Sequence / Name) as Original:
                    Size = Original.size
                Prediction = np.asarray(Image.fromarray(Probability[Index]).resize(Size, Image.Resampling.BILINEAR)) >= 0.5
                Predictions.append(Prediction)
                Frame_Numbers.append(int(Name.split('_')[0]))
                Relative = Sequence.relative_to(Args.Data_Root) if Args.Data_Root in Sequence.parents else Sequence
                Row = {'Sequence': str(Relative), 'Frame': Name}
                Mask_Path = Sequence / Name.replace('_img.png', '_mask_LV.png')
                if not Mask_Path.is_file():
                    Mask_Path = Sequence / Name.replace('_img.png', '_mask.png')
                if Mask_Path.is_file():
                    try:
                        with Image.open(Mask_Path) as Mask:
                            if Mask.size != Size:
                                raise ValueError('image and label sizes differ')
                            Target = np.asarray(Mask.convert('L')) >= 127
                        Row.update(Frame_Metrics(Prediction, Target))
                    except (OSError, ValueError) as Error:
                        Record_Skips([{'Stage': 'Mask', 'Path': str(Mask_Path), 'Reason': str(Error)}], Skip_File)
                Rows.append(Row)
            Temporal = Temporal_Metrics(Predictions, Frame_Numbers)
            for Row, NL in zip(Rows, Temporal):
                Row['NL'] = float(NL)
                for Name in Names:
                    Value = Row.get(Name, float('nan'))
                    if np.isfinite(Value):
                        Totals[Name] += Value
                        Counts[Name] += 1
                Writer.writerow(Row)
            if Counts['Dice']:
                Progress.set_postfix(Dice=f'{Totals["Dice"] / Counts["Dice"]:.4f}')
    Summary = {Name: {'Mean': Totals[Name] / Counts[Name] if Counts[Name] else None, 'Frames': Counts[Name]} for Name in Names}
    (Args.Output_Dir / 'summary.json').write_text(json.dumps(Summary, indent=2), encoding='utf-8')
    print(json.dumps(Summary, indent=2))


if __name__ == '__main__':
    Test(Parser.parse_args())
