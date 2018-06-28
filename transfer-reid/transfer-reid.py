from __future__ import print_function, absolute_import
import argparse
import os.path as osp


import sys
from gt_extractor import GtExtractor
from retrainer import Retrainer
import datasets
import os

sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/')
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/utils')


def main():#(args):

    path_to_basemodel = '/export/livia/data/FHafner/data/logdir/tum_depth/training/triplet_resnet18_tv/' \
                        'model_best.pth.tar'
    path_to_save_gt = '/export/livia/data/FHafner/data/logdir/tum_depth/training/triplet_resnet18_tv/'
    # path_to_basemodel = '//export/livia/data/FHafner/data/logdir/tum_depth/retrain/triplet_resnet18_control/' \
    #                     'model_best.pth.tar'
    # path_to_save_gt = '/export/livia/data/FHafner/data/logdir/tum_depth/retrain/triplet_resnet18_control/'




    # gtExtractor = GtExtractor(path_to_basemodel)
    # gtExtractor.extract_gt('tum_depth', path_to_save_gt=path_to_save_gt, extract_for='train')
    # gtExtractor.extract_gt('tum_depth', path_to_save_gt=path_to_save_gt, extract_for='val')
    # gtExtractor.extract_gt('tum_depth', path_to_save_gt=path_to_save_gt, extract_for='gallery')
    # gtExtractor.extract_gt('tum_depth', path_to_save_gt=path_to_save_gt, extract_for='query')

    path_to_retmodel = '/export/livia/data/FHafner/data/logdir/tum/retraining/triplet_resnet18_tv/'
    # path_to_retmodel = '/export/livia/data/FHafner/data/logdir/tum/training/triplet_resnet18_highreg/'


    if not os.path.exists(path_to_retmodel):
        os.makedirs(path_to_retmodel)

    retrainer = Retrainer(path_to_basemodel, dropout=0)
    retrainer.retrain('tum', 'tum_depth', path_to_save_gt, batch_size=64, epochs=200, combine_trainval=True,
                      workers=3, path_to_retmodel=path_to_retmodel)

    pass


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Softmax loss classification")
    # # data
    # parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
    #                     choices=datasets.names())
    # parser.add_argument('-b', '--batch-size', type=int, default=256)
    # parser.add_argument('-j', '--workers', type=int, default=4)
    # parser.add_argument('--split', type=int, default=0)
    # parser.add_argument('--height', type=int,
    #                     help="input height, default: 256 for resnet*, "
    #                          "144 for inception")
    # parser.add_argument('--width', type=int,
    #                     help="input width, default: 128 for resnet*, "
    #                          "56 for inception")
    # parser.add_argument('--combine-trainval', action='store_true',
    #                     help="train and val sets together for training, "
    #                          "val set alone for validation")
    # # model
    # parser.add_argument('-a', '--arch', type=str, default='resnet50',
    #                     choices=models.names())
    # parser.add_argument('--features', type=int, default=128)
    # parser.add_argument('--dropout', type=float, default=0.5)
    # # optimizer
    # parser.add_argument('--lr', type=float, default=0.1,
    #                     help="learning rate of new parameters, for pretrained "
    #                          "parameters it is 10 times smaller than this")
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--weight-decay', type=float, default=5e-4)
    # # training configs
    # parser.add_argument('--resume', type=str, default='', metavar='PATH')
    # parser.add_argument('--evaluate', action='store_true',
    #                     help="evaluation only")
    # parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--start_save', type=int, default=0,
    #                     help="start saving checkpoints after specific epoch")
    # parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--print-freq', type=int, default=1)
    # # metric learning
    # parser.add_argument('--dist-metric', type=str, default='euclidean',
    #                     choices=['euclidean', 'kissme'])
    # # misc
    # working_dir = osp.dirname(osp.abspath(__file__))
    # parser.add_argument('--data-dir', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, 'data'))
    # parser.add_argument('--logs-dir', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, 'logs'))
    # main(parser.parse_args())
    main()
