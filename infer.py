import argparse
import sys
from pathlib import Path

Release_Root = Path(__file__).resolve().parent
sys.path.insert(0, str(Release_Root.parent))

from Code_FOFSS_Echocardiographic_Sequence_Segmentation.dataset import Read_Paths, Sequence_Dataset, Sequence_Batch, Validate_Sequences
from Code_FOFSS_Echocardiographic_Sequence_Segmentation.model import FOFSS
from ignite.engine import Engine

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion
from torch.utils.data import DataLoader
from tqdm import tqdm

Parser = argparse.ArgumentParser(description='Run FOFSS and save original-resolution contour videos.')
Parser.add_argument('--Checkpoint', type=Path, default=Path('weights/Weight_FOFSS.pth'))
Parser.add_argument('--Test_File', type=Path, default=None, help='Sequence list relative to Data_Root, or an absolute path.')
Parser.add_argument('--Sequence', type=Path, default=None)
Parser.add_argument('--Data_Root', type=Path, default=None, help='Absolute path to the data directory.')
Parser.add_argument('--Output_Dir', type=Path, default=Path('predictions'))
Parser.add_argument('--Device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
Parser.add_argument('--Workers', type=int, default=4)
Parser.add_argument('--Progress_Width', type=int, default=160)
Parser.add_argument('--Video_FPS', type=int, default=10)


def Save_AVI(Video_Path, Sequence, Names, Probability, FPS):
    with Image.open(Sequence / Names[0]) as Original:
        Width, Height = Original.size
    Video_Path.parent.mkdir(parents=True, exist_ok=True)
    Video_Size = (Width + Width % 2, Height + Height % 2)
    Video = cv2.VideoWriter(str(Video_Path), cv2.VideoWriter_fourcc(*'MJPG'), FPS, Video_Size)
    if not Video.isOpened():
        raise OSError(f'Cannot open AVI writer: {Video_Path}')
    try:
        for Frame, Name in enumerate(Names):
            with Image.open(Sequence / Name) as Original:
                Gray = np.asarray(Original.convert('L').resize((Width, Height), Image.Resampling.BILINEAR))
            Restored = np.asarray(Image.fromarray(Probability[Frame]).resize((Width, Height), Image.Resampling.BILINEAR))
            Mask = Restored >= 0.5
            Boundary = Mask & ~binary_erosion(Mask, border_value=0)
            Picture = cv2.cvtColor(Gray, cv2.COLOR_GRAY2BGR)
            Picture[Boundary] = (0, 0, 255)
            Picture = cv2.copyMakeBorder(Picture, 0, Height % 2, 0, Width % 2, cv2.BORDER_REPLICATE)
            Video.write(Picture)
    finally:
        Video.release()


def Infer(Args):
    if Args.Data_Root is None or not Args.Data_Root.is_absolute():
        Parser.error('Set --Data_Root to the absolute path of your data directory.')
    Args.Checkpoint = (Release_Root / Args.Checkpoint).resolve()
    Args.Output_Dir = (Release_Root / Args.Output_Dir).resolve()
    Args.Data_Root = Args.Data_Root.resolve()
    Args.Test_File = (Args.Data_Root / Args.Test_File).resolve() if Args.Test_File else Args.Data_Root / 'Data_Split_txt/cross_validation_folds/dir_format/fold_1/test.txt'
    Args.Sequence = (Args.Data_Root / Args.Sequence).resolve() if Args.Sequence else None
    if Args.Video_FPS <= 0:
        Parser.error('Video_FPS must be positive.')
    if Args.Output_Dir == Args.Data_Root or Args.Data_Root in Args.Output_Dir.parents:
        Parser.error('Output_Dir must be outside Data_Root.')
    print(f'Checkpoint: {Args.Checkpoint}', flush=True)
    State = torch.load(Args.Checkpoint, map_location='cpu', weights_only=True)
    Device = torch.device(Args.Device)
    Model = FOFSS(**State['Model_Config']).to(Device)
    Model.load_state_dict(State['Model'], strict=True)
    Model.eval()
    Args.Output_Dir.mkdir(parents=True, exist_ok=True)
    print(f'Video output: {Args.Output_Dir / "videos"}', flush=True)
    Paths = [Args.Sequence.resolve()] if Args.Sequence is not None else Read_Paths(Args.Test_File, Args.Data_Root, Args.Progress_Width)
    Paths = Validate_Sequences(Paths, Description='Check input images', Progress_Width=Args.Progress_Width)
    Data = Sequence_Dataset(Paths, State['Image_Size'], Require_Masks=False)
    Loader = DataLoader(Data, batch_size=1, collate_fn=Sequence_Batch, num_workers=Args.Workers)
    Saved_Videos = 0

    def Infer_Sequence(Runner, Batch):
        nonlocal Saved_Videos
        for Skipped in Batch['Skipped']:
            tqdm.write(f'Skipped: {Skipped["Path"]}: {Skipped["Reason"]}')
        if Batch['Images'] is not None:
            Probability = Model(Batch['Images'].to(Device))['Probability'][0, :, 0].float().cpu().numpy()
            Sequence = Path(Batch['Paths'][0])
            Relative = Sequence.relative_to(Args.Data_Root) if Args.Data_Root in Sequence.parents else Path(Sequence.parent.name) / Sequence.name
            Video_Path = Args.Output_Dir / 'videos' / Relative.parent / (Relative.name + '_prediction.avi')
            Save_AVI(Video_Path, Sequence, Batch['Names'][0], Probability, Args.Video_FPS)
            Saved_Videos += 1
            if Saved_Videos == 1:
                tqdm.write(f'First video saved: {Video_Path}')
        Progress.update(1)

    with torch.no_grad(), tqdm(total=len(Loader), desc='Inference', ncols=Args.Progress_Width) as Progress:
        if Paths:
            Engine(Infer_Sequence).run(Loader, max_epochs=1)
    print(f'Inference complete: {Saved_Videos} videos saved in {Args.Output_Dir / "videos"}', flush=True)


if __name__ == '__main__':
    sys.stderr.reconfigure(line_buffering=True)
    Infer(Parser.parse_args())
