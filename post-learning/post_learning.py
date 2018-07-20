from __future__ import print_function, absolute_import

import argparse

import sys

sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/')
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/utils')
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/transfer-reid')

from gt_extractor import GtExtractor
from retrainer import Retrainer
import datasets
import os
import os.path as osp
import torch
from torch.utils.data import DataLoader
from post_trainer import PostTrainer
from post_utils.postpreprocessors import PreprocessorPost, PreprocessorPostEval
from tensorboardX import SummaryWriter


def get_data(path_to_gt1, path_to_gt2, batch_size=64, workers=4):

    train_loader = DataLoader(
        PreprocessorPost(osp.join(path_to_gt1, 'gt_train.txt'), osp.join(path_to_gt2, 'gt_train.txt')),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=False, drop_last=True)

    val_loader = DataLoader(
        PreprocessorPost(osp.join(path_to_gt1, 'gt_val_probe.txt'), osp.join(path_to_gt2, 'gt_val_gallery.txt')),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    val_probe_loader = DataLoader(
        PreprocessorPostEval(osp.join(path_to_gt1, 'gt_val_probe.txt')),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    val_gallery_loader = DataLoader(
        PreprocessorPostEval(osp.join(path_to_gt2, 'gt_val_gallery.txt')),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    test_probe_loader = DataLoader(
        PreprocessorPostEval(osp.join(path_to_gt1, 'gt_query.txt')),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    test_gallery_loader = DataLoader(
        PreprocessorPostEval(osp.join(path_to_gt2, 'gt_gallery.txt')),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return train_loader, val_loader, val_probe_loader, val_gallery_loader, test_probe_loader, test_gallery_loader





modal1 = 'sysu_ir'
modal2 = 'sysu'

logdir = '/export/livia/data/FHafner/data/logdir/'

path_to_save_gt1 = osp.join(logdir, modal1, 'train/triplet-resnet18/')
path_to_model1 = osp.join(path_to_save_gt1 + '/model_best.pth.tar')

path_to_save_gt2 = osp.join(logdir, modal2, 'train/triplet-resnet18/')
path_to_model2 = osp.join(path_to_save_gt2 + '/model_best.pth.tar')

name = 'test3/'
path_to_postmodel = osp.join(logdir, 'postmodel', name)

# gtExtractor = GtExtractor(path_to_model1)
# gtExtractor.extract_gt(modal1, path_to_save_gt=path_to_save_gt1, extract_for='train')
# gtExtractor.extract_gt(modal1, path_to_save_gt=path_to_save_gt1, extract_for='val_gallery')
# gtExtractor.extract_gt(modal1, path_to_save_gt=path_to_save_gt1, extract_for='val_probe')
# gtExtractor.extract_gt(modal1, path_to_save_gt=path_to_save_gt1, extract_for='gallery')
# gtExtractor.extract_gt(modal1, path_to_save_gt=path_to_save_gt1, extract_for='query')
# # #
# gtExtractor = GtExtractor(path_to_model2)
# gtExtractor.extract_gt(modal2, path_to_save_gt=path_to_save_gt2, extract_for='train')
# gtExtractor.extract_gt(modal2, path_to_save_gt=path_to_save_gt2, extract_for='val_gallery')
# gtExtractor.extract_gt(modal2, path_to_save_gt=path_to_save_gt2, extract_for='val_probe')
# gtExtractor.extract_gt(modal2, path_to_save_gt=path_to_save_gt2, extract_for='gallery')
# gtExtractor.extract_gt(modal2, path_to_save_gt=path_to_save_gt2, extract_for='query')

batch_size = 64
train_loader, val_loader, val_probe_loader, val_gallery_loader, test_probe_loader, test_gallery_loader = \
    get_data(path_to_save_gt1, path_to_save_gt2, batch_size)

epochs = 500

writer = SummaryWriter(path_to_postmodel + '/tensorboard')

post_trainer = PostTrainer(path_to_postmodel)

post_trainer.train(epochs, train_loader, val_loader, val_probe_loader, val_gallery_loader, test_probe_loader,
                   test_gallery_loader, writer)
