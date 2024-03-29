# Cross-Modal Distillation Network for Person Re-identification in RGB-Depth

Code for Paper: A Cross-Modal Distillation Network for Person Re-identification in RGB-Depth

The repository is based on open-reid (https://github.com/Cysu/open-reid). A big thanks to Cysu for providing an excellent repository as a baseline for research in re-identification. Open-Reid as the baseline is highly adjusted to the needs for cross-modal re-identification in this repository.
Hence, for the general structuring of the open-reid part please refer to Cysu's [Documentation](https://cysu.github.io/open-reid/notes/overview.html). 

The code in this repository is divided in two directories:
1. open-reid -> training of networks with softmax or triplet loss (softmax also with one-stream, two-stream, zero-padding).
2. transfer-reid -> performs the distillation step based on a network trained with open-reid.

## Environment to run the code
Please refer to requirements.yml. 

## Data 
Download routines for the datasets are not provided in the repository. Please download and prepare the datasets yourself according to our paper:

- [BIWI](http://robotics.dei.unipd.it/reid/index.php/8-dataset/2-overview-biwi)

- [RobotPKU](https://github.com/lianghu56/RobotPKU-RGBD-ID-dataset/blob/master/Dataset%20Downloads%20Dddress.md)

We included the split.json and meta.json in the datasets folder to facilitate repeating the experiments from our paper. Hence, you need to fill the images folder for each dataset accordingly. For more information refer to [Cysu's Documentation](https://cysu.github.io/open-reid/notes/overview.html).

## Reproducing the results
After preparing the datasets you can run the code as follows (one example is given for each algorithm sample). More details on the input arguments can be found in the files: 

### Softmax Loss Single-Modal
cd open-reid/examples

python softmax_loss.py -d biwi_depth -a resnet50 -batch-size 64 --epochs 100 --split 0 --features 128 --logs-dir PATH_TO_LOGDIR --data-dir PATH_TO_DATA_DIR
### Triplet Loss Single-Modal
cd open-reid/examples

python triplet_loss.py -d biwi_depth -a resnet50 -batch-size 64 --epochs 100 --split 0 --features 128 --logs-dir PATH_TO_LOGDIR --data-dir PATH_TO_DATA_DIR
### Softmax Loss One-Stream Network
cd open-reid/examples

python softmax_loss_onestream.py -d1 biwi -d2 biwi_depth -a resnet50 -batch-size 64 --epochs 100 --split 0 --features 128 --logs-dir PATH_TO_LOGDIR --data-dir PATH_TO_DATA_DIR
### Softmax Loss Two-Stream Network
cd open-reid/examples

python softmax_loss_twostream.py -d1 biwi -d2 biwi_depth -a resnet50 -batch-size 64 --epochs 100 --split 0 --features 128 --logs-dir PATH_TO_LOGDIR --data-dir PATH_TO_DATA_DIR
### Softmax Loss Zero-Padding Network
cd open-reid/examples

python softmax_loss_zp.py -d1 biwi -d2 biwi_depth -a resnet50 -batch-size 64 --epochs 100 --split 0 --features 128 --logs-dir PATH_TO_LOGDIR --data-dir PATH_TO_DATA_DIR


### Cross-Distillation Approach based on Model trained with Softmax Loss Single-Modal or Triplet Loss Single-Modal
cd transfer-reid

CUDA_VISIBLE_DEVICES=0 python transfer-reid.py -f biwi_depth -t biwi --path-to-orig PATH_TO_LOGDIR_OF_STEP_1 --name UNIQUE_NAME --split-id 0 --logdir PATH_TO_LOGDIR --data-dir PATH_TO_DATA_DIR --batch-size 16 --extract True

## Reference
Hafner, F. M., Bhuiyan, A., Kooij, J. F., & Granger, E. (2019, September). RGB-depth cross-modal person re-identification. In 2019 16th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS) (pp. 1-8). IEEE.

