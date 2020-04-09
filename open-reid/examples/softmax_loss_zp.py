from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import sys
sys.path.append('../reid')
sys.path.append('../utils')

import datasets
import models
from dist_metric import DistanceMetric
from loss import TripletLoss
from trainers import Trainer
from evaluators import Evaluator
from utils.data import transforms as T
from utils.data.preprocessor import PreprocessorZP
from utils.data.sampler import RandomIdentitySampler
from utils.logging import Logger
from utils.serialization import load_checkpoint, save_checkpoint
from tensorboardX import SummaryWriter
import os


def get_data(name1, name2, split_id, data_dir, height, width, batch_size, workers,
             combine_trainval):
    root1 = osp.join(data_dir, name1)
    root2 = osp.join(data_dir, name2)

    dataset1 = datasets.create(name1, root1, split_id=split_id)
    dataset2 = datasets.create(name2, root2, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set1 = dataset1.trainval if combine_trainval else dataset1.train
    num_classes1 = (dataset1.num_trainval_ids if combine_trainval
                   else dataset1.num_train_ids)

    train_set2 = dataset2.trainval if combine_trainval else dataset2.train
    num_classes2 = (dataset2.num_trainval_ids if combine_trainval
                   else dataset2.num_train_ids)

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        PreprocessorZP(train_set1, train_set2, root=dataset1.images_dir, root2=dataset2.images_dir,
                              transform=train_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=False, drop_last=True)

    val_loader1 = DataLoader(
        PreprocessorZP(list(set(dataset1.val_probe) | set(dataset1.val_gallery)), root=dataset1.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    val_loader2 = DataLoader(
        PreprocessorZP(list(set(dataset2.val_probe) | set(dataset2.val_gallery)), root=dataset2.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    test_loader1 = DataLoader(
        PreprocessorZP(list(set(dataset1.query) | set(dataset1.gallery)),
                     root=dataset1.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    test_loader2 = DataLoader(
        PreprocessorZP(list(set(dataset2.query) | set(dataset2.gallery)),
                     root=dataset2.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return dataset1, dataset2, num_classes1, train_loader, val_loader1, val_loader2, test_loader1, test_loader2


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    use_all = True
    torch.set_num_threads(1)


    name_val = 'zp' + args.dataset1 + '-' + args.dataset2 + '-' + args.logs_dir.split('/')[-1] + '-split' + str(args.split) + '-val'
    name_test = 'zp' + args.dataset1 + '-' + args.dataset2 + '-' + args.logs_dir.split('/')[-1] + '-split' + str(args.split) + '-test'
    print(name_test)

    print(args)

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)

    # args.dataset is dataset name; split = ?; data_dir is position of data; height,width is
    # how input looks like; batch_size; workers?; combine trainval gives better accuracy
    dataset1, dataset2, num_classes, train_loader, val_loader1, val_loader2, test_loader1, test_loader2 = \
        get_data(args.dataset1, args.dataset2, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers,
                 args.combine_trainval)

    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes)


    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    # model = nn.DataParallel(model).cuda()

    # writer for summary
    logs_dir_tb = args.logs_dir + '/tensorboard/'
    if not os.path.exists(logs_dir_tb):
        os.makedirs(logs_dir_tb)

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        # metric.train(model, train_loader)
        # save_to = '/export/livia/data/FHafner/data/logdir/sysu_onestream/test1/'
        # evaluator.evaluate_all_and_save_sysu(test_loader1, test_loader2, save_to, height=args.height, width=args.width)
        model.num_classes = 0

        print("Validation: ")
        evaluator.evaluate_cm(val_loader1, val_loader2, dataset1.val_probe, dataset1.val_gallery,
                              dataset2.val_probe, dataset2.val_gallery, 10, writer=None, epoch=None,
                              metric=None, calc_cmc=True, use_all=use_all, final=name_val)


        print("Test:")
        evaluator.evaluate_cm(test_loader1, test_loader2, dataset1.query, dataset1.gallery,
                              dataset2.query, dataset2.gallery, 10, writer=None, epoch=None,
                              metric=None, calc_cmc=True, use_all=use_all, test=True, final=name_test)
        model.num_classes = num_classes

        # evaluator.evaluate_single_shot(dataset1.val_probe, dataset1.val_probe, 0, None, 0,
        #                                root=osp.join(args.data_dir, args.dataset1), height=args.height,
        #                      width=args.width, name2save="Internal Re-ID Validation Dataset " + args.dataset1, evaluations=5)
        # evaluator.evaluate_single_shot(dataset2.val_probe, dataset2.val_probe, 0, None, 0,
        #                                root=osp.join(args.data_dir, args.dataset2), height=args.height,
        #                      width=args.width, name2save="Internal Re-ID Validation Dataset " + args.dataset2, evaluations=5)
        #
        # evaluator.evaluate_single_shot_cm(dataset1.val_probe, dataset2.val_gallery, 0, None, 0,
        #                                   root1 = osp.join(args.data_dir, args.dataset1), height = args.height,
        #                                   width = args.width, root2 = osp.join(args.data_dir, args.dataset2),
        #                                   name2save = "Cross modality val 1")
        # evaluator.evaluate_single_shot_cm(dataset2.val_probe, dataset1.val_gallery, 0, None, 0,
        #                                   root1 = osp.join(args.data_dir, args.dataset2), height = args.height,
        #                                   width = args.width, root2 = osp.join(args.data_dir, args.dataset1),
        #                                   name2save = "Cross modality val 1")

        # evaluator.evaluate_single_shot(dataset1.query, dataset1.gallery, 0, None, 0,
        #                                root=osp.join(args.data_dir, args.dataset1), height=args.height,
        #                      width=args.width, name2save="Internal Re-ID Test set " + args.dataset1, evaluations=5)
        #
        # evaluator.evaluate_single_shot(dataset2.query, dataset2.gallery, 0, None, 0,
        #                                root=osp.join(args.data_dir, args.dataset2), height=args.height,
        #                      width=args.width, name2save="Internal Re-ID Test set " + args.dataset2, evaluations=5)
        #
        # evaluator.evaluate_single_shot_cm(dataset1.query, dataset2.gallery, 0, None, 0,
        #                                   root1 = osp.join(args.data_dir, args.dataset1), height = args.height,
        #                                   width = args.width, root2 = osp.join(args.data_dir, args.dataset2),
        #                                   name2save = "Cross modality test 1 ")
        # evaluator.evaluate_single_shot_cm(dataset2.query, dataset1.gallery, 0, None, 0,
        #                                   root1 = osp.join(args.data_dir, args.dataset2), height = args.height,
        #                                   width = args.width, root2 = osp.join(args.data_dir, args.dataset1),
        #                                   name2save = "Cross modality test 1 ")

        # evaluator.evaluate_all_and_save_sysu(test_loader2, test_loader1, args.logs_dir, height=args.height,
        #                                      width=args.width)

        return

    # writer for summary
    writer = SummaryWriter(logs_dir_tb)

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()

    param_groups = model.parameters()

    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Trainer
    trainer = Trainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = 60 if args.arch == 'inception' else 40
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # evaluator.evaluate_single_shot(dataset1.val_probe, dataset1.val_probe, 0, writer, 0,
    #                                root=osp.join(args.data_dir, args.dataset1), height=args.height,
    #                                width=args.width, name2save="Internal Re-ID Validation Dataset " + args.dataset1,
    #                                evaluations=5)
    # evaluator.evaluate_single_shot(dataset2.val_probe, dataset2.val_probe, 0, writer, 0,
    #                                root=osp.join(args.data_dir, args.dataset2), height=args.height,
    #                                width=args.width, name2save="Internal Re-ID Validation Dataset " + args.dataset2,
    #                                evaluations=5)
    #
    # evaluator.evaluate_single_shot_cm(dataset1.val_probe, dataset2.val_gallery, 0, writer, 0,
    #                                   root1=osp.join(args.data_dir, args.dataset1), height=args.height,
    #                                   width=args.width, root2=osp.join(args.data_dir, args.dataset2),
    #                                   name2save="Cross modality val 1")
    # evaluator.evaluate_single_shot_cm(dataset2.val_probe, dataset1.val_gallery, 0, writer, 0,
    #                                   root1=osp.join(args.data_dir, args.dataset2), height=args.height,
    #                                   width=args.width, root2=osp.join(args.data_dir, args.dataset1),
    #                                   name2save="Cross modality val 1")

    evaluator.evaluate_cm(val_loader1, val_loader2, dataset1.val_probe, dataset1.val_gallery,
                          dataset2.val_probe, dataset2.val_gallery, 10, writer=writer, epoch=None,
                          metric=None, calc_cmc=True, use_all=use_all)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch+1, train_loader, optimizer, args.print_freq, writer)
        if epoch < args.start_save:
            continue

        if epoch % 10 == 0 or top1 == 0:
            model.num_classes = 0

            top1 = evaluator.evaluate_cm(val_loader1, val_loader2, dataset1.val_probe, dataset1.val_gallery,
                                  dataset2.val_probe, dataset2.val_gallery, 10, writer=writer, epoch=epoch+1,
                                  metric=None, calc_cmc=True, use_all=use_all)

            model.num_classes = num_classes

            is_best = top1 > best_top1
            best_top1 = max(top1, best_top1)
            print("is_best: " + str(is_best))
            if is_best:
                print("epoch: " + str(epoch))
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_top1': best_top1,
                }, is_best, fpath=osp.join(args.logs_dir, 'model_best.pth.tar'))

            print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, top1, best_top1, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    model.num_classes = 0

    evaluator.evaluate_cm(val_loader1, val_loader2, dataset1.val_probe, dataset1.val_gallery,
                                 dataset2.val_probe, dataset2.val_gallery, 10, writer=None, epoch=None,
                                 metric=None, calc_cmc=True, use_all=use_all, final=name_val)

    evaluator.evaluate_cm(test_loader1, test_loader2, dataset1.query, dataset1.gallery,
                                 dataset2.query, dataset2.gallery, 10, writer=None, epoch=None,
                                 metric=None, calc_cmc=True, use_all=use_all, test=True, final=name_test)
    model.num_classes = num_classes



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d1', '--dataset1', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-d2', '--dataset2', type=str, default='cuhk03',
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
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=50)
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
                        default='/export/livia/data/FHafner/data')#osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
