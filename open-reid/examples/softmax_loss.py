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
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/')
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/utils')

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
import os


def get_data(name, split_id, data_dir, height, width, batch_size, workers,
             combine_trainval, zero_pad):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    if zero_pad:
        train_transformer = T.Compose([
            T.RandomSizedRectCropDepth(height, width),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
        ])

        test_transformer = T.Compose([
            T.RectScaleDepth(height, width),
            T.ToTensor(),
            normalizer,
        ])

    else:
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
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(dataset.val_probe+dataset.val_gallery, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return dataset, num_classes, train_loader, val_loader, test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    name_val = args.dataset + '-' + args.logs_dir.split('/')[-1] + '-split' + str(args.split) + '-val'
    name_test = args.dataset + '-' + args.logs_dir.split('/')[-1] + '-split' + str(args.split) + '-test'
    print(name_test)

    use_all = True
    # writer for summary


    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)

    # args.dataset is dataset name; split = ?; data_dir is position of data; height,width is
    # how input looks like; batch_size; workers?; combine trainval gives better accuracy
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers,
                 args.combine_trainval, args.zero_pad)

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

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        # evaluator.evaluate(val_loader, dataset.val_probe, dataset.val_gallery, 1, writer=None, epoch=None,
        #                    metric=None, calc_cmc=False, use_all=use_all)
        # evaluator.make_comp(test_loader, dataset.query, dataset.gallery, 1, writer=None, epoch=None, metric=None,
        #                    calc_cmc=True, use_all=use_all)
        model.num_classes = 0
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, 1, writer=None, epoch=None, metric=None,
                           calc_cmc=True, use_all=use_all)
        model.num_classes = num_classes

        return

    # writer for summary
    logs_dir_tb = args.logs_dir + '/tensorboard/'
    if not os.path.exists(logs_dir_tb):
        os.makedirs(logs_dir_tb)

    # if os.listdir(logs_dir_tb) != []:
    #     raise Exception('There is already a trained model in the directory!')

    writer = SummaryWriter(logs_dir_tb)

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)


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

    model.num_classes = 0
    top1 = evaluator.evaluate(val_loader, dataset.val_probe, dataset.val_gallery, args.print_freq, writer, 0,
                              metric=None, calc_cmc=True, use_all=use_all)
    model.num_classes = num_classes

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer, args.print_freq, writer)
        if epoch < args.start_save:
            continue

        if epoch % 10 == 0:
            model.num_classes = 0
            top1 = evaluator.evaluate(val_loader, dataset.val_probe, dataset.val_gallery, args.print_freq, writer, epoch,
                                      metric=None, calc_cmc=True, use_all=use_all)

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
            model.num_classes = num_classes

    # Final test
    print('Test with best model:')
    model.num_classes = 0
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(val_loader, dataset.val_probe, dataset.val_gallery, 1, writer=None, epoch=None, metric=None,
                       calc_cmc=True, use_all=use_all, final=name_val)
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, 1, writer=None, epoch=None, metric=None,
                       calc_cmc=True, use_all=use_all, final=name_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
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
    parser.add_argument('--zero-pad', type=bool, default=False)
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
