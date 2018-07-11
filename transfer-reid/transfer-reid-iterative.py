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
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/examples')

from triplet_loss_func import triplet_loss_func
import datasets
import models
from dist_metric import DistanceMetric
from loss import TripletLoss
from trainers import Trainer
from evaluators import Evaluator
from utils.data import transforms as T
from utils.data.preprocessor import Preprocessor
from utils.data.sampler import RandomIdentitySampler
from utils.logging import Logger
from utils.serialization import load_checkpoint, save_checkpoint
from tensorboardX import SummaryWriter


def main(args):

    iterations = 6
    # for iterations. if it is even the same as in the beginning will be evaluated, if odd other way round
    epochs_triplet = 40
    epochs_retrain = 10
    name = 'resnet18_40_10/'


    from_ = 'sysu_ir'
    to_ = 'sysu'

    logdir = '/export/livia/data/FHafner/data/logdir/'

    # TODO: path to save gt is in the moment saving on top
    path_to_save_gt = logdir + '/path_to_gt/'
    path_to_origmodel = logdir + from_ + '/train/triplet-resnet18/model_best.pth.tar'
    path_to_retsavemodel = logdir + to_ + '/retraining/' + name + str(0) + '/'
    path_to_retstartmodel = logdir + to_ + '/train/triplet-resnet18/model_best.pth.tar'

    if os.path.exists(path_to_retsavemodel):
        raise Exception('There is already a trained model in the directory!')

    for itera in range(iterations):

        gtExtractor = GtExtractor(path_to_origmodel)
        gtExtractor.extract_gt_av(from_, to_, path_to_save_gt=path_to_save_gt, extract_for='train')
        gtExtractor.extract_gt_av(from_, to_, path_to_save_gt=path_to_save_gt, extract_for='val')
        gtExtractor.extract_gt_av(from_, to_, path_to_save_gt=path_to_save_gt, extract_for='query')

        # gtExtractor.extract_gt(from_, path_to_save_gt=path_to_save_gt, extract_for='train')
        # gtExtractor.extract_gt(from_, path_to_save_gt=path_to_save_gt, extract_for='val_gallery')
        # gtExtractor.extract_gt(from_, path_to_save_gt=path_to_save_gt, extract_for='val_probe')
        # gtExtractor.extract_gt(from_, path_to_save_gt=path_to_save_gt, extract_for='gallery')
        # gtExtractor.extract_gt(from_, path_to_save_gt=path_to_save_gt, extract_for='query')

        if not os.path.exists(path_to_retsavemodel):
            os.makedirs(path_to_retsavemodel)

        retrainer = Retrainer(path_to_retstartmodel, path_to_origmodel, dropout=0.3, freeze_model=True)

        retrainer.retrain(to_, from_, path_to_save_gt, batch_size=64, epochs=epochs_retrain, combine_trainval=False, workers=3,
                          path_to_retmodel=path_to_retsavemodel)
        # retrainer.re_evaluate_retrain(to_, from_, path_to_save_gt, batch_size=64, combine_trainval=False,
        #                            workers=3, path_to_retsavemodel=path_to_save_gt)

        args.dataset = to_
        args.arch = 'resnet18'
        args.logs_dir = path_to_retsavemodel
        args.resume = path_to_retsavemodel + '/model_best.pth.tar'
        args.epochs = epochs_triplet
        triplet_loss_func(args)

        if itera != iterations-1:
            to_, from_ = from_, to_

            path_to_origmodel, path_to_retstartmodel = path_to_retsavemodel, path_to_origmodel
            path_to_origmodel += '/model_best.pth.tar'
            path_to_retsavemodel = logdir + to_ + '/retraining/' + name + str(itera+1)

    # final transfer training
    retrainer = Retrainer(path_to_retstartmodel, path_to_origmodel, dropout=0.3, freeze_model=True)

    retrainer.retrain(to_, from_, path_to_save_gt, batch_size=64, epochs=epochs_retrain, combine_trainval=False,
                      workers=3,
                      path_to_retmodel=path_to_retsavemodel)
    retrainer.re_evaluate_retrain(to_, from_, path_to_save_gt, batch_size=64, combine_trainval=False,
                                  workers=3, path_to_retsavemodel=path_to_save_gt)


    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triplet loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0002,
                        help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--evaluate-cm', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/export/livia/data/FHafner/data')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())