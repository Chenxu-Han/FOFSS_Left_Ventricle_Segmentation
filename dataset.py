import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.signal import find_peaks
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, default_collate
from tqdm import tqdm


def Read_Paths(Split_File, Data_Root, Progress_Width=160):
    Data_Root = Path(Data_Root).resolve()
    print(f'Read sequence list: {Split_File}', flush=True)
    Paths = []
    Lines = Path(Split_File).read_text(encoding='utf-8-sig').splitlines()
    for Line in tqdm(Lines, desc='Read paths', unit='seq', ncols=Progress_Width, miniters=1):
        Line = Line.strip().replace('\\', '/')
        if Line:
            if '/Data_Png_All/' in '/' + Line:
                Path_Value = Data_Root / 'Data_Png_All' / Line.split('Data_Png_All/', 1)[1]
            elif Path(Line).is_absolute():
                Path_Value = Path(Line)
            else:
                Path_Value = Data_Root / Line
            Paths.append(Path_Value.resolve())
    Missing = {}
    for Index, Sequence in enumerate(tqdm(Paths, desc='Check directories', unit='seq', ncols=Progress_Width, miniters=1)):
        if not Sequence.is_dir() and 'Data_Png_All' in Sequence.parts:
            Patient_Root = Sequence.parents[2]
            if not Patient_Root.is_dir():
                continue
            Key = (Sequence.parts[-4], Sequence.parts[-2], Sequence.name)
            Group = Missing.setdefault(Patient_Root, {})
            Entry = Group.setdefault(Key, {'Indices': [], 'Matches': []})
            Entry['Indices'].append(Index)
    Used = set(Paths)
    if Missing:
        with tqdm(desc='Find renamed views', unit='dir', ncols=Progress_Width, miniters=1) as Progress:
            for Patient_Root, Group in Missing.items():
                Progress.set_postfix(Patient=Patient_Root.name, refresh=True)
                for Directory, _, _ in os.walk(Patient_Root):
                    Candidate = Path(Directory)
                    Key = (Candidate.parts[-4], Candidate.parts[-2], Candidate.name)
                    if Key in Group:
                        Group[Key]['Matches'].append(Candidate)
                    Progress.update(1)
                for Entry in Group.values():
                    if len(Entry['Indices']) == 1 and len(Entry['Matches']) == 1:
                        Index, Candidate = Entry['Indices'][0], Entry['Matches'][0]
                        if Candidate not in Used:
                            Paths[Index] = Candidate
                            Used.add(Candidate)
    return Paths


def Patient_Id(Sequence):
    Parts = Sequence.parts
    Dataset_Name = Parts[Parts.index('Data_Png_All') + 1]
    return Dataset_Name + '/' + Sequence.parents[2].name


def Split_Training(Paths, Validation_Fraction, Seed):
    Patients = sorted({Patient_Id(Path_Value) for Path_Value in Paths})
    random.Random(Seed).shuffle(Patients)
    Validation_Count = max(1, round(len(Patients) * Validation_Fraction))
    Validation_Patients = set(Patients[:Validation_Count])
    Train = [Path_Value for Path_Value in Paths if Patient_Id(Path_Value) not in Validation_Patients]
    Validation = [Path_Value for Path_Value in Paths if Patient_Id(Path_Value) in Validation_Patients]
    return Train, Validation


def Record_Skips(Problems, Skip_File):
    if Problems and Skip_File is not None:
        Skip_File = Path(Skip_File)
        New_File = not Skip_File.is_file()
        with Skip_File.open('a', encoding='utf-8', newline='') as File:
            Writer = csv.DictWriter(File, fieldnames=['Stage', 'Path', 'Reason'])
            if New_File:
                Writer.writeheader()
            Writer.writerows(Problems)


def Validate_Sequences(Paths, Require_Masks=False, Skip_File=None, Description='Check sequences', Progress_Width=160):
    Valid_Paths, Problems = [], []
    for Sequence in tqdm(Paths, desc=Description, unit='seq', ncols=Progress_Width, miniters=1):
        if not Sequence.is_dir():
            Problems.append({'Stage': 'Check', 'Path': str(Sequence), 'Reason': 'directory missing'})
            continue
        Reasons = []
        Image_Paths = list(Sequence.glob('*_img.png'))
        Mask_Paths = (list(Sequence.glob('*_mask_LV.png')) or list(Sequence.glob('*_mask.png'))) if Require_Masks else []
        if not Image_Paths:
            Reasons.append('no *_img.png frames')
        if Require_Masks and not Mask_Paths:
            Reasons.append('no *_mask_LV.png or *_mask.png labels')
        for Mask_Path in Mask_Paths:
            Image_Path = Sequence / (Mask_Path.name.split('_')[0] + '_img.png')
            if not Image_Path.is_file():
                Reasons.append(f'missing paired image {Image_Path.name} for {Mask_Path.name}')
        if Reasons:
            Problems.append({'Stage': 'Check', 'Path': str(Sequence), 'Reason': '; '.join(Reasons)})
        else:
            Valid_Paths.append(Sequence)
    Record_Skips(Problems, Skip_File)
    print(f'{Description}: {len(Valid_Paths)} usable; {len(Problems)} skipped.', flush=True)
    return Valid_Paths


def Cycle_Phase(Areas, Frame_Numbers):
    Phase = torch.zeros(len(Frame_Numbers))
    Phase_Valid = torch.zeros(len(Frame_Numbers), dtype=torch.bool)
    Area = np.asarray(Areas, dtype=np.float64)
    Frame_Numbers = np.asarray(Frame_Numbers)
    Indices = np.flatnonzero(Area > 0)
    Breaks = np.flatnonzero(np.diff(Frame_Numbers[Indices]) != 1) + 1
    for Segment in np.split(Indices, Breaks):
        if len(Segment) < 5:
            continue
        Curve = Area[Segment]
        Spread = np.percentile(Curve, 95) - np.percentile(Curve, 5)
        if Spread <= 0:
            continue
        _, Peaks = find_peaks(Curve, prominence=0.5 * Spread, plateau_size=(None, None))
        Boundaries = Segment[Peaks['right_edges']]
        for Start, End in zip(Boundaries[:-1], Boundaries[1:]):
            if min(Area[Start], Area[End]) - Area[Start:End + 1].min() < 0.5 * Spread:
                continue
            Phase[Start:End + 1] = torch.from_numpy((Frame_Numbers[Start:End + 1] - Frame_Numbers[Start]) / (Frame_Numbers[End] - Frame_Numbers[Start]))
            Phase_Valid[Start:End + 1] = True
    return Phase, Phase_Valid


class Sequence_Dataset(Dataset):
    def __init__(self, Paths, Image_Size=128, Tokenizer=None, Text_Length=512, Require_Masks=False, Skip_File=None, Progress_Width=160):
        self.Paths = Paths
        self.Image_Size = Image_Size
        self.Require_Masks = Require_Masks or Tokenizer is not None
        self.Tokens = []
        if Tokenizer is not None:
            self.Paths = []
            for Sequence in tqdm(Paths, desc='Read and tokenize reports', unit='seq', ncols=Progress_Width, miniters=1):
                Patient = Sequence.parents[1]
                if not (Patient / 'metric.txt').is_file() and not (Patient / 'report.txt').is_file():
                    Patient = Sequence.parents[2]
                Metric_Path, Report_Path = Patient / 'metric.txt', Patient / 'report.txt'
                try:
                    Metric = json.loads(Metric_Path.read_text(encoding='utf-8-sig')) if Metric_Path.is_file() else {}
                    Metric = {Name: Value for Name, Value in Metric.items() if Value != 0.0}
                    Report = Report_Path.read_text(encoding='utf-8-sig') if Report_Path.is_file() else ''
                    Text = json.dumps(Metric, ensure_ascii=False) + '\n' + Report
                    Text = Text.replace('[SEP]', '').replace('\t', ' ').replace('"', '').replace(':', ' ')
                except (OSError, ValueError, AttributeError, TypeError) as Error:
                    Record_Skips([{'Stage': 'Text', 'Path': str(Sequence), 'Reason': str(Error)}], Skip_File)
                    continue
                Tokens = Tokenizer(Text, return_tensors='pt', padding='max_length', truncation=True, max_length=Text_Length)
                self.Tokens.append({Name: Value[0] for Name, Value in Tokens.items()})
                self.Paths.append(Sequence)
            print(f'Reports ready: {len(self.Paths)} sequences; {len(Paths) - len(self.Paths)} skipped.', flush=True)

    def __len__(self):
        return len(self.Paths)

    def __getitem__(self, Index):
        Sequence = self.Paths[Index]
        Frames, Masks, Original_Masks, Areas, Sizes = [], [], [], [], []
        try:
            Mask_Paths = (list(Sequence.glob('*_mask_LV.png')) or list(Sequence.glob('*_mask.png'))) if self.Require_Masks else []
            if self.Require_Masks and not Mask_Paths:
                return {'Path': str(Sequence), 'Error': 'no segmentation masks'}
            Mask_Paths = sorted(Mask_Paths, key=lambda Item: int(Item.name.split('_')[0]))
            Image_Paths = [Sequence / (Item.name.split('_')[0] + '_img.png') for Item in Mask_Paths]
            if not Mask_Paths:
                Image_Paths = sorted(Sequence.glob('*_img.png'), key=lambda Item: int(Item.name.split('_')[0]))
            if not Image_Paths:
                return {'Path': str(Sequence), 'Error': 'no readable image frames'}
            for Image_Path in Image_Paths:
                with Image.open(Image_Path) as Frame:
                    Sizes.append(Frame.size)
                    Frame = Frame.convert('L').resize((self.Image_Size, self.Image_Size), Image.Resampling.BILINEAR)
                    Frames.append(torch.from_numpy(np.array(Frame, dtype=np.float32) / 255.0)[None])
            for Frame_Index, Mask_Path in enumerate(Mask_Paths):
                with Image.open(Mask_Path) as Mask:
                    if Mask.size != Sizes[Frame_Index]:
                        return {'Path': str(Sequence), 'Error': f'image and label sizes differ: {Mask_Path.name}'}
                    Mask = Mask.convert('L')
                    Original_Masks.append(np.array(Mask) >= 127)
                    Areas.append(np.count_nonzero(Original_Masks[-1]))
                    Mask = Mask.resize((self.Image_Size, self.Image_Size), Image.Resampling.NEAREST)
                    Masks.append(torch.from_numpy((np.array(Mask) >= 127).astype(np.float32))[None])
        except (OSError, ValueError) as Error:
            return {'Path': str(Sequence), 'Error': str(Error)}
        Frame_Numbers = [int(Item.name.split('_')[0]) for Item in Image_Paths]
        Phase, Phase_Valid = Cycle_Phase(Areas, Frame_Numbers) if Masks else (torch.zeros(len(Frames)), torch.zeros(len(Frames), dtype=torch.bool))
        return {
            'Images': torch.stack(Frames), 'Masks': torch.stack(Masks) if Masks else None,
            'Original_Masks': Original_Masks, 'Phase': Phase, 'Phase_Valid': Phase_Valid,
            'Tokens': self.Tokens[Index] if self.Tokens else None,
            'Path': str(Sequence), 'Names': [Item.name for Item in Image_Paths],
        }


def Sequence_Batch(Items):
    Skipped = [{'Stage': 'Load', 'Path': Item['Path'], 'Reason': Item['Error']} for Item in Items if 'Error' in Item]
    Items = [Item for Item in Items if 'Error' not in Item]
    if not Items:
        return {'Images': None, 'Skipped': Skipped}
    Lengths = torch.tensor([len(Item['Images']) for Item in Items])
    Images = pad_sequence([Item['Images'] for Item in Items], batch_first=True)
    Masks = None
    if Items[0]['Masks'] is not None:
        Masks = pad_sequence([Item['Masks'] for Item in Items], batch_first=True)
    Tokens = None
    if Items[0]['Tokens'] is not None:
        Tokens = default_collate([Item['Tokens'] for Item in Items])
    Phase = pad_sequence([Item['Phase'] for Item in Items], batch_first=True)
    Phase_Valid = pad_sequence([Item['Phase_Valid'] for Item in Items], batch_first=True)
    return {
        'Images': Images, 'Masks': Masks, 'Tokens': Tokens, 'Skipped': Skipped,
        'Original_Masks': [Item['Original_Masks'] for Item in Items],
        'Phase': Phase, 'Phase_Valid': Phase_Valid,
        'Valid': torch.arange(Images.shape[1])[None] < Lengths[:, None],
        'Paths': [Item['Path'] for Item in Items], 'Names': [Item['Names'] for Item in Items],
    }
