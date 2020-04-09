# Cross-Modal Distillation Network for Person Re-identification in RGB-Depth

Code for Paper: A Cross-Modal Distillation Network for Person Re-identification in RGB-Depth

The repository is based on open-reid (https://github.com/Cysu/open-reid). A big thanks to Cysu for providing an excellent repository as a baseline for research in re-identification. Open-Reid as the baseline is highly adjusted to the needs for cross-modal re-identification in this repository.
Hence, for the general structuring of the open-reid part please refer to his [Documentation](https://cysu.github.io/open-reid/notes/overview.html). 

The code in this repository is divided in two directories:
1. open-reid -> training of networks with softmax or triplet loss 
2. transfer-reid -> performs the distillation step based on a network trained with open-reid

## Environment to run the code
Please refer to requirements. 

## Reproducing the results


## Data 
Unfortunately we cannot provide the data of the BIWI and RobotPKU datasets directly.
BIWI: 
RobotPKU

Transfer-reid needed for cross-modal distillation

Usage of available Examples:

CUDA_VISIBLE_DEVICES=0 python softmax_loss.py -d biwi_depth -a resnet50 -batch-size 64 --epochs 100 --workers 2 --start_save 0 --print-freq 10 --split 0 --features 512 --logs-dir PATH_TO_LOGDIR --data-dir PATH_TO_DATA_DIR

CUDA_VISIBLE_DEVICES=0 python /export/livia/home/vision/FHafner/masterthesis/transfer-reid/transfer-reid.py -f biwi_depth  -t biwi --path-to-orig PATH_TO_LOGDIR/logdir/train/softmax-resnet50-split0/ --name softmax-resnet50-dim512-split0/ --split-id 0 --extract True

