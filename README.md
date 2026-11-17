# FOFSS

PyTorch implementation of the fractional-order fuzzy state-space model for left-ventricular segmentation in echocardiographic sequences.

## Installation

Use Python 3.8 and PyTorch 2.4.1. Install [Git LFS](https://git-lfs.com/) before cloning:

```bash
git clone https://github.com/Chenxu-Han/FOFSS_Left_Ventricle_Segmentation.git Code_FOFSS_Echocardiographic_Sequence_Segmentation
cd Code_FOFSS_Echocardiographic_Sequence_Segmentation
git lfs pull
```

For CUDA 12.1, install the matching PyTorch build before the other dependencies:

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --Data_Root "$DATA_ROOT"
```

Training uses `fold_1/train.txt` and splits its patients into training and validation subsets. Use `--Train_File` and `--Val_File` to supply separate lists. Settings are available through `python train.py --help`.

Checkpoints are saved in `runs/train_fofss/`: `best.pth` for inference and `last.pth` for training recovery. Training resumes from `last.pth` automatically; use `--Resume` to select another checkpoint, or `--Resume ""` for a fresh run.

## Inference

```bash
python infer.py --Data_Root "$DATA_ROOT" --Checkpoint weights/Weight_FOFSS.pth
```

Use `--Sequence` for a single sequence or `--Test_File` for a sequence list. Inference writes AVI videos with red contours to `predictions/videos/`. `--Output_Dir` changes the output location. Relative checkpoint and output paths are resolved from this project directory.

## Weights

Git LFS provides the FOFSS checkpoint in `weights/Weight_FOFSS.pth` and the pretrained [Google MobileBERT](https://huggingface.co/google/mobilebert-uncased) files in `weights/mobilebert-uncased/`. MobileBERT is used only during training.

## Citation

Chenxu Han et al. *Exploring Ambiguous Temporal Dynamics of Latent Cardiac Motion: Fractional-Order Fuzzy State-Space Modeling for Echocardiographic Sequence Segmentation.*
