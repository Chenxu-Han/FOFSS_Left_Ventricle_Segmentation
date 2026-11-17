import argparse
import csv
import json
import sys
import os
import random
from pathlib import Path

Release_Root = Path(__file__).resolve().parent
sys.path.insert(0, str(Release_Root.parent))

if __name__ == '__main__':
    sys.stderr.reconfigure(line_buffering=True)
    print('Starting FOFSS: loading dependencies ...', flush=True)

from Code_FOFSS_Echocardiographic_Sequence_Segmentation.dataset import Read_Paths, Sequence_Dataset, Sequence_Batch, Split_Training, Validate_Sequences, Record_Skips, Patient_Id
from Code_FOFSS_Echocardiographic_Sequence_Segmentation.loss import Training_Loss, Segmentation_Loss
from Code_FOFSS_Echocardiographic_Sequence_Segmentation.metrics import Dice_Score
from Code_FOFSS_Echocardiographic_Sequence_Segmentation.model import FOFSS

from ignite.engine import Engine
from ignite.handlers import DiskSaver

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from transformers import MobileBertModel, MobileBertTokenizerFast

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
if __name__ == '__main__':
    print('Dependencies loaded. Detecting CUDA devices ...', flush=True)
Parser = argparse.ArgumentParser(description='Train FOFSS on complete echocardiographic sequences.')
Parser.add_argument('--Data_Root', type=Path, default=None, help='Absolute path to the data directory.')
Parser.add_argument('--Train_File', type=Path, default=None, help='Sequence list; relative paths are resolved under Data_Root.')
Parser.add_argument('--Val_File', type=Path, default=None, help='Optional independent validation list.')
Parser.add_argument('--Text_Model', type=Path, default=Path('weights/mobilebert-uncased'))
Parser.add_argument('--Run_Dir', type=Path, default=Path('runs/train_fofss'))
Parser.add_argument('--Resume', default='auto')
Parser.add_argument('--Device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
Parser.add_argument('--Epochs', type=int, default=100)
Parser.add_argument('--Patience', type=int, default=15)
Parser.add_argument('--Batch_Size', type=int, default=8)
Parser.add_argument('--Accumulation', type=int, default=16)
Parser.add_argument('--Learning_Rate', type=float, default=1e-3)
Parser.add_argument('--Image_Size', type=int, default=128)
Parser.add_argument('--Hidden_Dim', type=int, default=512)
Parser.add_argument('--Rule_Count', type=int, default=12)
Parser.add_argument('--History_Length', type=int, default=100)
Parser.add_argument('--Text_Length', type=int, default=512)
Parser.add_argument('--Frame_Batch', type=int, default=8)
Parser.add_argument('--Validation_Fraction', type=float, default=0.2)
Parser.add_argument('--Seed', type=int, default=42)
Parser.add_argument('--Workers', type=int, default=4)
Parser.add_argument('--Save_Every', type=int, default=50)
Parser.add_argument('--Progress_Width', type=int, default=250)
Parser.set_defaults(Decay_Epochs=10, Decay_Factor=0.5, Time_Step=1.0, Alpha_Mean=0.184, Delta_Mean=0.023, Alpha_Concentration=10.0, Delta_Std=1.0, A_Std=1.0, B_Std=1.0, Text_Variance=1.0, Text_Weight=1.0, Aux_Weight=1.0, Occupancy_Weight=1.0)


class Text_Encoder(nn.Module):
    def __init__(self, Model_Path, Hidden_Dim):
        super().__init__()
        self.Bert, Loading_Info = MobileBertModel.from_pretrained(str(Model_Path), local_files_only=True, output_loading_info=True)
        if Loading_Info['missing_keys'] or Loading_Info['mismatched_keys'] or Loading_Info['error_msgs'] or any(not Name.startswith('cls.') for Name in Loading_Info['unexpected_keys']):
            raise RuntimeError(f'MobileBERT checkpoint does not match the text backbone: {Loading_Info}')
        self.Bert.requires_grad_(False)
        self.Projection = nn.Sequential(nn.LayerNorm(self.Bert.config.hidden_size), nn.Linear(self.Bert.config.hidden_size, Hidden_Dim))
        nn.init.xavier_uniform_(self.Projection[-1].weight)
        nn.init.zeros_(self.Projection[-1].bias)

    def forward(self, Tokens):
        self.Bert.eval()
        with torch.no_grad():
            Features = self.Bert(**Tokens).last_hidden_state[:, 0]
        return self.Projection(Features)


def Train(Args):
    if Args.Data_Root is None or not Args.Data_Root.is_absolute():
        Parser.error('Set --Data_Root to the absolute path of your data directory.')
    Args.Data_Root = Args.Data_Root.resolve()
    Split_Root = Args.Data_Root / 'Data_Split_txt/cross_validation_folds/dir_format/fold_1'
    Args.Train_File = (Args.Data_Root / Args.Train_File).resolve() if Args.Train_File else Split_Root / 'train.txt'
    Args.Val_File = (Args.Data_Root / Args.Val_File).resolve() if Args.Val_File else None
    if min(Args.Batch_Size, Args.Accumulation, Args.Save_Every, Args.Frame_Batch, Args.History_Length) < 1 or Args.Workers < 0:
        Parser.error('Batch sizes, history length and checkpoint interval must be positive; Workers must be nonnegative.')
    if not 0 < Args.Validation_Fraction < 1:
        Parser.error('Validation_Fraction must be between 0 and 1.')
    for Name in ['Text_Model', 'Run_Dir']:
        setattr(Args, Name, (Release_Root / getattr(Args, Name)).resolve())
    if Args.Resume and Args.Resume != 'auto':
        Args.Resume = str((Release_Root / Args.Resume).resolve())
    print(f'Run directory: {Args.Run_Dir}', flush=True)
    Checkpoint_Writer = DiskSaver(Args.Run_Dir, atomic=True, require_empty=False)
    Resume_State = None
    if Args.Resume == 'auto':
        Last_Checkpoint = Args.Run_Dir / 'last.pth'
        if Last_Checkpoint.is_file():
            print(f'Load checkpoint: {Last_Checkpoint}', flush=True)
            Resume_State = torch.load(Last_Checkpoint, map_location='cpu', weights_only=True)
    elif Args.Resume:
        print(f'Load checkpoint: {Args.Resume}', flush=True)
        Resume_State = torch.load(Args.Resume, map_location='cpu', weights_only=True)
    print('Checkpoint loading complete.', flush=True)
    Warm_Start = Resume_State is not None and 'Optimizer' not in Resume_State
    if Resume_State is not None and not Warm_Start:
        Keep = {'Run_Dir', 'Resume', 'Device', 'Workers', 'Epochs', 'Patience', 'Progress_Width', 'Train_File', 'Val_File', 'Data_Root', 'Text_Model', 'Save_Every'}
        for Name, Value in Resume_State['Train_Args'].items():
            if Name not in Keep:
                setattr(Args, Name, Value)
    if Warm_Start:
        for Name, Value in Resume_State['Model_Config'].items():
            setattr(Args, Name, Value)
        Args.Image_Size = Resume_State['Image_Size']
    Args.Min_Occupancy = 0.5 / Args.Rule_Count
    Args.Dice_Weight = float(Args.Image_Size ** 2)

    random.seed(Args.Seed)
    np.random.seed(Args.Seed)
    torch.manual_seed(Args.Seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    Device = torch.device(Args.Device)
    Paths = Read_Paths(Args.Train_File, Args.Data_Root, Args.Progress_Width)
    if Args.Val_File is None:
        print(f'Split {len(Paths)} sequences by patient ...', flush=True)
        Train_Paths, Val_Paths = Split_Training(Paths, Args.Validation_Fraction, Args.Seed)
    else:
        Train_Paths = Paths
        Val_Paths = Read_Paths(Args.Val_File, Args.Data_Root, Args.Progress_Width)
        if {Patient_Id(Path_Value) for Path_Value in Train_Paths} & {Patient_Id(Path_Value) for Path_Value in Val_Paths}:
            Parser.error('Training and validation lists contain overlapping patients.')
    Train_Paths = Validate_Sequences(Train_Paths, True, Args.Run_Dir / 'train_skipped.csv', Description='Check training data', Progress_Width=Args.Progress_Width)
    Val_Paths = Validate_Sequences(Val_Paths, True, Args.Run_Dir / 'val_skipped.csv', Description='Check validation data', Progress_Width=Args.Progress_Width)
    if not Train_Paths or not Val_Paths:
        print('No usable training or validation sequences; see skipped CSV files.')
        return
    print(f'Load local tokenizer: {Args.Text_Model}', flush=True)
    Tokenizer = MobileBertTokenizerFast.from_pretrained(str(Args.Text_Model), local_files_only=True)
    Train_Data = Sequence_Dataset(Train_Paths, Args.Image_Size, Tokenizer, Args.Text_Length, Require_Masks=True, Skip_File=Args.Run_Dir / 'train_skipped.csv', Progress_Width=Args.Progress_Width)
    Val_Data = Sequence_Dataset(Val_Paths, Args.Image_Size, Require_Masks=True)
    Data_Paths = {'Train': list(map(str, Train_Data.Paths)), 'Validation': list(map(str, Val_Data.Paths))}
    for Name, Values in [('train', Train_Data.Paths), ('val', Val_Data.Paths)]:
        (Args.Run_Dir / (Name + '.txt')).write_text('\n'.join(map(str, Values)) + '\n', encoding='utf-8')
    if not len(Train_Data):
        print('No usable training reports; see train_skipped.csv.')
        return
    Val_Loader = DataLoader(Val_Data, batch_size=Args.Batch_Size, collate_fn=Sequence_Batch, num_workers=Args.Workers, generator=torch.Generator().manual_seed(Args.Seed))
    Model_Config = {Name: getattr(Args, Name) for Name in ['Hidden_Dim', 'Rule_Count', 'History_Length', 'Time_Step', 'Alpha_Mean', 'Delta_Mean', 'Frame_Batch']}
    print(f'Initialize FOFSS and move to {Device} ...', flush=True)
    Model = FOFSS(**Model_Config).to(Device)
    print(f'Load local MobileBERT weights and move to {Device} ...', flush=True)
    Text_Model = Text_Encoder(Args.Text_Model, Args.Hidden_Dim).to(Device)
    print('Models ready.', flush=True)
    if Warm_Start:
        Model.load_state_dict(Resume_State['Model'], strict=True)
        if 'Text_Projection' in Resume_State:
            Text_Model.Projection.load_state_dict(Resume_State['Text_Projection'], strict=True)
        Resume_State = None
        print('Loaded model weights; training starts with a new optimizer at epoch 1.')
    Parameters = list(Model.parameters()) + list(Text_Model.Projection.parameters())
    Optimizer = torch.optim.Adam(Parameters, lr=Args.Learning_Rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=Args.Decay_Epochs, gamma=Args.Decay_Factor)
    Start_Epoch, Start_Batch, Global_Batch, Best_Dice = 1, 0, 0, -1.0
    Bad_Epochs = 0
    Sums, Count, Group_Count = {}, 0, 0
    if Resume_State is not None:
        print('Restore model, optimizer and training position ...', flush=True)
        Model.load_state_dict(Resume_State['Model'], strict=True)
        Text_Model.Projection.load_state_dict(Resume_State['Text_Projection'], strict=True)
        Optimizer.load_state_dict(Resume_State['Optimizer'])
        Scheduler.load_state_dict(Resume_State['Scheduler'])
        Start_Epoch, Start_Batch = Resume_State['Epoch'], Resume_State['Batch']
        Global_Batch, Best_Dice = Resume_State['Global_Batch'], Resume_State['Best_Dice']
        Bad_Epochs = Resume_State['Bad_Epochs']
        random.setstate(Resume_State['Random_State'])
        if 'Numpy_State' in Resume_State:
            Numpy_State = Resume_State['Numpy_State']
            np.random.set_state((Numpy_State[0], Numpy_State[1].numpy().astype(np.uint32), *Numpy_State[2:]))
        torch.set_rng_state(Resume_State['Torch_State'])
        if Device.type == 'cuda' and Resume_State['Cuda_State'] is not None:
            torch.cuda.set_rng_state(Resume_State['Cuda_State'], Device)
        if all(Resume_State.get('Data_Paths', {}).get(Name) == Values for Name, Values in Data_Paths.items()):
            Sums, Count = Resume_State['Sums'], Resume_State['Count']
            Group_Count = Resume_State['Group_Count']
            for Parameter, Gradient in zip(Parameters, Resume_State['Gradients']):
                Parameter.grad = Gradient.to(Device) if Gradient is not None else None
        else:
            Start_Batch = 0
            Best_Dice = -1.0
            Bad_Epochs = 0
            print(f'Data list changed or absent in checkpoint: keep weights and optimizer; restart epoch {Start_Epoch}.')
        if Resume_State.get('Dice_Resolution') != 'original':
            Start_Batch, Best_Dice, Bad_Epochs = 0, -1.0, 0
            Sums, Count, Group_Count = {}, 0, 0
            Optimizer.zero_grad(set_to_none=True)
            print('Dice now uses original labels; restart this epoch and reset validation selection statistics.')
    if Bad_Epochs >= Args.Patience:
        print(f'The saved run already reached the stopping criterion ({Bad_Epochs} epochs without improvement).')
        return

    Train_Args = {Name: str(Value) if isinstance(Value, Path) else Value for Name, Value in vars(Args).items()}
    (Args.Run_Dir / 'config.json').write_text(json.dumps(Train_Args, indent=2), encoding='utf-8')
    print(f'Train: {len(Train_Data)} sequences; validation: {len(Val_Data)} sequences; test.txt reserved for final testing.', flush=True)
    print(f'Effective batch: {Args.Batch_Size * Args.Accumulation} sequences; output: {Args.Run_Dir}', flush=True)

    def Save_State(Name, Epoch, Batch):
        Numpy_State = np.random.get_state()
        State = {'Model': Model.state_dict(), 'Model_Config': Model_Config, 'Image_Size': Args.Image_Size, 'Dice_Resolution': 'original', 'Bad_Epochs': Bad_Epochs, 'Text_Projection': Text_Model.Projection.state_dict(), 'Train_Args': Train_Args, 'Optimizer': Optimizer.state_dict(), 'Scheduler': Scheduler.state_dict(), 'Epoch': Epoch, 'Batch': Batch, 'Global_Batch': Global_Batch, 'Best_Dice': Best_Dice, 'Data_Paths': Data_Paths, 'Group_Count': Group_Count, 'Sums': Sums, 'Count': Count, 'Random_State': random.getstate(), 'Torch_State': torch.get_rng_state(), 'Cuda_State': torch.cuda.get_rng_state(Device) if Device.type == 'cuda' else None, 'Gradients': [
            Parameter.grad for Parameter in Parameters]}
        State['Numpy_State'] = (Numpy_State[0], torch.tensor(Numpy_State[1].astype(np.int64)), *Numpy_State[2:])
        Checkpoint_Writer(State, Name)

    def Train_Batch(Runner, Data):
        nonlocal Global_Batch, Count, Group_Count
        Batch = Runner.state.iteration
        if Epoch == Start_Epoch and Batch <= Start_Batch:
            Progress.set_postfix(Resume=f'{Batch}/{Start_Batch}', refresh=False)
            Progress.update(1)
            return
        Record_Skips(Data['Skipped'], Args.Run_Dir / 'train_skipped.csv')
        if Data['Images'] is not None:
            if Batch == 1:
                tqdm.write(f'First training batch loaded: {len(Data["Images"])} sequences, '
                           f'{int(Data["Valid"].sum())} real frames; starting forward/backward.', file=sys.stderr)
            Images, Masks, Valid = (Data[Name].to(Device) for Name in ['Images', 'Masks', 'Valid'])
            Tokens = {Name: Value.to(Device) for Name, Value in Data['Tokens'].items()}
            Phase, Phase_Valid = Data['Phase'].to(Device), Data['Phase_Valid'].to(Device)
            Output = Model(Images, Masks)
            Text_Features = Text_Model(Tokens)
            Loss, Parts = Training_Loss(Output, Masks, Valid, Text_Features, Model, Args, Phase, Phase_Valid)
            (Loss * len(Images)).backward()
            Group_Count += len(Images)
            Dice_Sum, Frame_Count = Dice_Score(Output['Probability'], Data['Original_Masks'], Valid)
            Parts['Loss'] = Loss.item()
            for Name, Value in Parts.items():
                Sums[Name] = Sums.get(Name, 0.0) + Value * len(Images)
            Sums['Dice'] = Sums.get('Dice', 0.0) + Dice_Sum
            Sums['Frames'] = Sums.get('Frames', 0) + Frame_Count
            Sums['Phase_Frames'] = Sums.get('Phase_Frames', 0) + int((Valid & Phase_Valid).sum())
            Count += len(Images)
            Progress.set_postfix(Loss=f'{Loss.item():.4f}', Dice=f'{Sums["Dice"] / Sums["Frames"]:.4f}', refresh=False)
        if Group_Count and (Batch % Args.Accumulation == 0 or Batch == len(Loader)):
            for Parameter in Parameters:
                if Parameter.grad is not None:
                    Parameter.grad.div_(Group_Count)
            Optimizer.step()
            Model.State.Project_Stable()
            Optimizer.zero_grad(set_to_none=True)
            Group_Count = 0
        Global_Batch += 1
        if Global_Batch % Args.Save_Every == 0:
            Save_State('last.pth', Epoch, Batch)
        Progress.update(1)

    def Validate_Batch(Runner, Data):
        nonlocal Val_Dice, Val_Count, Val_Loss, Val_Samples, With_Images
        Record_Skips(Data['Skipped'], Args.Run_Dir / 'val_skipped.csv')
        if Data['Images'] is None:
            Validation_Progress.update(1)
            return
        Images, Masks, Valid = (Data[Name].to(Device) for Name in ['Images', 'Masks', 'Valid'])
        Probability = Model(Images)['Probability']
        Cross_Entropy, Dice_Loss = Segmentation_Loss(Probability, Masks, Valid)
        Dice_Sum, Frame_Count = Dice_Score(Probability, Data['Original_Masks'], Valid)
        Val_Dice += Dice_Sum
        Val_Count += Frame_Count
        Val_Loss += (Cross_Entropy + Args.Dice_Weight * Dice_Loss).item() * len(Images) / Args.Image_Size ** 2
        Val_Samples += len(Images)
        Validation_Progress.set_postfix(Loss=f'{Val_Loss / Val_Samples:.4f}', Dice=f'{Val_Dice / Val_Count:.4f}', refresh=False)
        if random.randrange(Val_Count) < Frame_Count:
            Sample, Frame = random.choice(Valid.nonzero().tolist())
            With_Images = torch.cat([Images[Sample, Frame, 0], Masks[Sample, Frame, 0], Probability[Sample, Frame, 0]], dim=1).cpu().numpy()
        Validation_Progress.update(1)

    Trainer, Evaluator = Engine(Train_Batch), Engine(Validate_Batch)

    for Epoch in range(Start_Epoch, Args.Epochs + 1):
        Model.train()
        Epoch_Learning_Rate = Optimizer.param_groups[0]['lr']
        Generator = torch.Generator().manual_seed(Args.Seed + Epoch)
        Loader = DataLoader(Train_Data, batch_size=Args.Batch_Size, shuffle=True, collate_fn=Sequence_Batch, num_workers=Args.Workers, generator=Generator)
        if Epoch != Start_Epoch or Start_Batch == 0:
            Sums, Count, Group_Count = {}, 0, 0
            Optimizer.zero_grad(set_to_none=True)
        Progress = tqdm(total=len(Loader), desc=f'Train {Epoch}/{Args.Epochs}', ncols=Args.Progress_Width)
        print(f'Training epoch {Epoch}: waiting for data loader ({Args.Workers} workers) ...', flush=True)
        Trainer.run(Loader, max_epochs=1)
        Progress.close()
        if not Count:
            print('No readable training sequences in this epoch; see train_skipped.csv.')
            return
        Scheduler.step()
        Model.eval()
        Val_Dice, Val_Count, Val_Loss, Val_Samples = 0.0, 0, 0.0, 0
        With_Images = None
        Validation_Progress = tqdm(total=len(Val_Loader), desc=f'Validate {Epoch}', ncols=Args.Progress_Width)
        with torch.no_grad():
            Evaluator.run(Val_Loader, max_epochs=1)
        Validation_Progress.close()
        Val_Dice = Val_Dice / Val_Count if Val_Count else float('nan')
        Weights = {'Model': Model.state_dict(), 'Model_Config': Model_Config, 'Image_Size': Args.Image_Size, 'Dice_Resolution': 'original', 'Epoch': Epoch, 'Validation_Dice': Val_Dice}
        if Val_Dice > Best_Dice:
            Best_Dice = Val_Dice
            Bad_Epochs = 0
            Checkpoint_Writer(Weights, 'best.pth')
        else:
            Bad_Epochs += 1
        if With_Images is not None:
            Image.fromarray((With_Images * 255).astype(np.uint8)).save(Args.Run_Dir / f'epoch_{Epoch:03d}_validation.png')
        Row = {'Epoch': Epoch, 'LR': Epoch_Learning_Rate, **{'Train_' + Name: Value / Count for Name, Value in Sums.items() if Name not in {'Dice', 'Frames', 'Phase_Frames'}}, 'Train_Frames': Sums['Frames'], 'Phase_Frames': Sums[
            'Phase_Frames'], 'Phase_Coverage': Sums['Phase_Frames'] / Sums['Frames'], 'Train_Dice': Sums['Dice'] / Sums['Frames'], 'Validation_Loss': Val_Loss / Val_Samples if Val_Samples else float('nan'), 'Validation_Dice': Val_Dice}
        Csv_Path = Args.Run_Dir / 'metrics.csv'
        Previous_Rows = []
        if Csv_Path.is_file():
            with Csv_Path.open(encoding='utf-8', newline='') as File:
                Previous_Rows = [Item for Item in csv.DictReader(File) if int(Item['Epoch']) != Epoch]
        Csv_Temporary = Csv_Path.with_suffix('.writing')
        with Csv_Temporary.open('w', newline='', encoding='utf-8') as File:
            Writer = csv.DictWriter(File, fieldnames=list(Row))
            Writer.writeheader()
            Writer.writerows(Previous_Rows)
            Writer.writerow(Row)
        Csv_Temporary.replace(Csv_Path)
        print(f'Epoch {Epoch}: validation Dice={Val_Dice:.6f}, best={Best_Dice:.6f}, phase frames={Sums["Phase_Frames"]}/{Sums["Frames"]}')
        Sums, Count = {}, 0
        Save_State('last.pth', Epoch + 1, 0)
        if Bad_Epochs >= Args.Patience:
            print(f'Stopped after {Bad_Epochs} epochs without improvement in validation Dice.')
            break


if __name__ == '__main__':
    Train(Parser.parse_args())
