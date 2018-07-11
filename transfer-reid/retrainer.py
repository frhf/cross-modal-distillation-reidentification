from __future__ import print_function, absolute_import

import torch
from torchvision import transforms
from torch.autograd import Variable

from torch.utils.data import DataLoader


from PIL import Image
import sys
import models
from torch import nn
import datasets
import models
from dist_metric import DistanceMetric
from loss import TripletLoss
from trainers import TrainerRetrainer
from evaluators import Evaluator
from utils.data import transforms as T
from utils.data.preprocessor import PreprocessorRetrain
from utils.data.preprocessor import Preprocessor
import copy
from utils.data.sampler import RandomIdentitySampler
from utils.logging import Logger
from utils.serialization import load_checkpoint, save_checkpoint
import pickle
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid')
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/utils')
from tensorboardX import SummaryWriter
import os.path as osp
import os
from collections import OrderedDict


class Retrainer:
    def __init__(self, path_to_model, path_to_origmodel=None, dropout=0, freeze_model=False):
        self.path_to_model = path_to_model
        self.dropout = dropout
        self.freeze_model = freeze_model
        self.model = self.load_model(self.path_to_model)
        if path_to_origmodel is None:
            self.model_orig = copy.deepcopy(self.model)
            self.model_orig.eval()
        else:
            self.path_to_origmodel = path_to_origmodel
            self.model_orig = self.load_model(self.path_to_origmodel)
            self.model_orig.eval()

    def load_model(self, path_to_model):
        name_dict = {
            110: 'resnet18',
            190: 'resnet34',
            203: 'inception',
            275: 'resnet50',
        }

        checkpoint = torch.load(path_to_model)
        self.model_arch = name_dict[len(checkpoint['state_dict'])]
        self.num_features = checkpoint['state_dict']['classifier.weight'].shape[1]
        self.num_classes = checkpoint['state_dict']['classifier.weight'].shape[0]
        model = models.create(self.model_arch, num_features=self.num_features,
                          dropout=self.dropout, num_classes=self.num_classes)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()

        if self.freeze_model:
            for params in model.parameters():
                if len(params) == 256:
                    break
                else:
                    params.requires_grad = False

        return model

    def retrain(self, dataset_ret, dataset_orig, path_to_gt, root='/export/livia/data/FHafner/data/',
                epochs=50, split_id=0, start_epoch=1, batch_size=64, workers=2,
                combine_trainval=False, path_to_retmodel=None):

        epochs += 1

        if self.model_arch == 'inception':
            height, width = (144, 56)
        else:
            height, width = (256, 128)

        # load retrain dataset
        dataset, train_loader, val_loader_ret, val_loader_int, test_loader, test_loader_ret = \
            get_data(dataset_ret, split_id, root, height, width, batch_size, workers, combine_trainval, path_to_gt)

        dataset_orig_, train_loader_orig_, val_loader_ret_orig_, val_loader_int_orig_, test_loader_orig_, \
        test_loader_ret_orig_ = \
            get_data(dataset_orig, split_id, root, height, width, batch_size, workers, combine_trainval, path_to_gt)

        if not os.path.exists(path_to_retmodel + '/tensorboard'):
            os.makedirs(path_to_retmodel + '/tensorboard')
        # initialize tensorboard writing
        writer = SummaryWriter(path_to_retmodel + '/tensorboard')

        self.model.cuda()
        self.model.train()

        criterion = nn.MSELoss(size_average=False).cuda()

        param_groups = self.model.parameters()

        lr = 0.00002
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                    momentum=0.9,
                                    weight_decay=0.0005,
                                    nesterov=True)

        metric = DistanceMetric(algorithm='euclidean')

        trainer = TrainerRetrainer(self.model, criterion)

        evaluator = Evaluator(self.model, self.model_orig)

        best_top1 = 0

        def adjust_lr(epoch, lr):
            if epoch > 100:
                lr * (0.001 ** ((epoch - 100) / 50.0))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

        evaluator.evaluate_single_shot(dataset.val_probe, dataset.val_gallery, 1, writer,
                                              0, root + dataset_ret, height, width,
                                              name2save="Reid in Retrain Domain: ")

        evaluator.evaluate_single_shot(dataset.train, dataset.train, 1, writer, 0, root + dataset_ret, height,
                                       width, name2save="Training data: ")

        evaluator.evaluate_one(dataset.train, dataset.train, 1, writer, 0, root + dataset_ret, height, width,
                                root2=root + dataset_orig)
        # evaluate CM
        top1 = evaluator.evaluate_single_shot_cm(dataset.val_probe, dataset_orig_.val_gallery, 1, writer, 0,
                                          root + dataset_ret, height, width, root + dataset_orig, 'Cross_modal: ')

        # if epoch % 4 == 0:
        evaluator.evaluate_validationloss(val_loader_ret, val_loader_int, criterion, 0, dataset.gallery, dataset.query,
                                       writer=writer)


        for epoch in range(start_epoch, epochs):
            adjust_lr(epoch, lr)

            trainer.train(epoch, train_loader, optimizer, print_freq=50, writer=writer)


            # if epoch % 10 == 0:
            #evaluate in retrained domain
            evaluator.evaluate_single_shot(dataset.val_probe, dataset.val_gallery, 1, writer,
                                                  epoch, root + dataset_ret, height, width,
                                                  name2save="Reid in Retrain Domain: ")

            evaluator.evaluate_single_shot(dataset.train, dataset.train, 1, writer, epoch, root + dataset_ret, height,
                                           width, name2save="Training data: ")

            evaluator.evaluate_one(dataset.train, dataset.train, 1, writer, epoch, root + dataset_ret, height, width,
                                   root2=root + dataset_orig)
            # evaluate CM
            top1 = evaluator.evaluate_single_shot_cm(dataset.val_probe, dataset_orig_.val_gallery, 1, writer, epoch,
                                              root + dataset_ret, height, width, root + dataset_orig, 'Cross_modal: ')

            # if epoch % 4 == 0:
            evaluator.evaluate_validationloss(val_loader_ret, val_loader_int, criterion, epoch, dataset.gallery, dataset.query,
                                       writer=writer)

            ## save model if its best
            is_best = top1 > best_top1
            best_top1 = max(top1, best_top1)
            if is_best:
                print("Model is saved.")
                save_checkpoint({
                    'state_dict': self.model.state_dict(),
                    'epoch': epoch + 1,
                    'best_top1': best_top1,
                }, is_best, fpath=osp.join(path_to_retmodel, 'model_best.pth.tar'))

    def re_evaluate_retrain(self, dataset_ret, dataset_orig, path_to_gt, root='/export/livia/data/FHafner/data/',
                         split_id=0, batch_size=64, workers=2, combine_trainval=False, path_to_retsavemodel=None):

        if self.model_arch == 'inception':
            height, width = (144, 56)
        else:
            height, width = (256, 128)

        # load retrain dataset
        dataset, train_loader, val_loader_ret, val_loader_int, test_loader, test_loader_ret = \
            get_data(dataset_ret, split_id, root, height, width, batch_size, workers, combine_trainval,
                     path_to_gt)

        dataset_orig_, train_loader_orig_, val_loader_ret_orig_, val_loader_int_orig_, test_loader_orig_, \
        test_loader_ret_orig_ = \
            get_data(dataset_orig, split_id, root, height, width, batch_size, workers, combine_trainval,
                     path_to_gt)

        evaluator = Evaluator(self.model, self.model_orig)
        top1 = evaluator.evaluate_single_shot_cm(dataset.query, dataset_orig_.gallery, 1, None, 0,
                                          root + dataset_ret, height, width, root + dataset_orig, 'Cross_modal test set: ')

        if dataset_ret == 'sysu' or dataset_ret == 'sysu_ir':
            evaluator.evaluate_all_and_save_sysu(test_loader, test_loader_orig_, path_to_retsavemodel, height, width)


def get_data(name, split_id, data_dir, height, width, batch_size, workers, combine_trainval, path_to_gt):

    root = data_dir +'/'+ name

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

    if name == 'tum' or name == 'tum_depth':
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

    train_set = pickle.load(open(path_to_gt + 'gt_train.txt', "rb"))
    val_query_set = pickle.load(open(path_to_gt + 'gt_val_probe.txt', "rb"))
    val_gallery_set = pickle.load(open(path_to_gt + 'gt_val_gallery.txt', "rb"))
    query_set = pickle.load(open(path_to_gt + 'gt_query.txt', "rb"))
    gallery_set = pickle.load(open(path_to_gt + 'gt_gallery.txt', "rb"))

    if val_gallery_set[0][0]!=val_query_set[0][0]:
        val_set = val_gallery_set + val_query_set
    else:
        val_set = val_gallery_set


    if query_set[0][0]!=gallery_set[0][0]:
        test_set = query_set + gallery_set
    else:
        test_set = gallery_set



    if combine_trainval:
        train_s = train_set + val_query_set
    else:
        train_s = train_set

    train_loader = DataLoader(
        PreprocessorRetrain(train_s, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=False, drop_last=False)

    val_loader_ret = DataLoader(
        PreprocessorRetrain(val_set, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    val_loader_int = DataLoader(
        Preprocessor(dataset.val_probe, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    test_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    # test_loader_g = DataLoader(
    #     Preprocessor(dataset.gallery,
    #                  root=root2, transform=test_transformer),
    #     batch_size=batch_size, num_workers=workers,
    #     shuffle=False, pin_memory=False)

    test_loader_ret = DataLoader(
        PreprocessorRetrain(test_set,
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return dataset, train_loader, val_loader_ret, val_loader_int, test_loader, test_loader_ret




