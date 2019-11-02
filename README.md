# A Cross-Modal Distillation Network for Person Re-identification in RGB-Depth

Code for Paper: A Cross-Modal Distillation Network for Person Re-identification in RGB-Depth

The base open-reid as from Cysu (https://github.com/Cysu/open-reid) but highly adjusted to the needs for cross-modal re-identification.

Transfer-reid needed for cross-modal distillation

Example usage:

CUDA_VISIBLE_DEVICES=0 python /cross-modal-distillation-network/open-reid/examples/softmax_loss.py -d biwi_depth -a resnet50 -b 64 --epochs 100 -j 2 --logs-dir PATH_TO_LOGDIR/logdir/biwi_depth/train/softmax-resnet50-split0 --start_save 0 --print-freq 10 --split 0 --features 512

CUDA_VISIBLE_DEVICES=0 python /export/livia/home/vision/FHafner/masterthesis/transfer-reid/transfer-reid.py -f biwi_depth  -t biwi --path-to-orig PATH_TO_LOGDIR/logdir/train/softmax-resnet50-split0/ --name softmax-resnet50-dim512-split0/ --split-id 0 --extract True
