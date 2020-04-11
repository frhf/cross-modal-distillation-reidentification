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
    def __init__(self, path_to_model, path_to_origmodel=None, dropout=0, freeze_model=True, load_weights=True):
        self.path_to_model = path_to_model
        self.dropout = dropout
        self.freeze_model = freeze_model
        self.model = self.load_model(self.path_to_model, load_weights)

        if path_to_origmodel is None:

            self.model_orig = self.load_model(self.path_to_model, load_weights=True)
            self.model_orig.eval()
        else:
            self.path_to_origmodel = path_to_origmodel
            self.model_orig = self.load_model(self.path_to_origmodel)
            self.model_orig.eval()

    def load_model(self, path_to_model, load_weights):
        name_dict = {
            110: 'resnet18',
            190: 'resnet34',
            203: 'inception',
            275: 'resnet50',
        }

        checkpoint = torch.load(path_to_model)
        self.model_arch = name_dict[len(checkpoint['state_dict'])]
        self.num_features = checkpoint['state_dict']['classifier.weight'].shape[1]
        self.num_classes_load = checkpoint['state_dict']['classifier.weight'].shape[0]
	self.num_classes_infer = checkpoint['num_classes']

        model = models.create(self.model_arch, num_features=self.num_features,
                          dropout=self.dropout, num_classes=self.num_classes_load)

        # hack to make softmax loss work
	if self.num_classes_load != self.num_classes_infer:
	    model.num_classes = 0 

        if load_weights:
            model.load_state_dict(checkpoint['state_dict'])
        model.cuda()

        # local variable to freeze later parts of model
        freeze_local = False
        if self.freeze_model:
            for params in model.parameters():
                if self.model_arch == 'resnet18' and len(params) == 256:
                    freeze_local = True
                if self.model_arch == 'resnet50' and len(params) == 1024:
                    freeze_local = True
                if freeze_local:
                    params.requires_grad = False

        return model

    def retrain(self, dataset_ret, dataset_orig, path_to_gt, root='../../data/',
                epochs=50, split_id=0, start_epoch=1, batch_size=64, workers=0,
                combine_trainval=False, path_to_retmodel=None, evaluate=None, name_val=[],
                name_test=[]):

        epochs += 1

        use_all = True


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

        if evaluate:
            
            model_load = models.create(self.model_arch, num_features=self.num_features,
                                  dropout=self.dropout, num_classes=self.num_classes_load)
            checkpoint = load_checkpoint(osp.join(path_to_retmodel, 'model_best.pth.tar'))
            model_load.load_state_dict(checkpoint['state_dict'])

	    if self.num_classes_load != self.num_classes_infer:
	       model_load.num_classes = 0 

            print(checkpoint['epoch'])
            evaluator = Evaluator(model_load, self.model_orig)

            print("Modality 1: " + val_loader_int.dataset.root)
            print("Modality 2: " + val_loader_int_orig_.dataset.root)

            print("Test:")
            evaluator.evaluate_cm(test_loader, test_loader_orig_, dataset.query, dataset.gallery,
                                  dataset_orig_.query, dataset_orig_.gallery, 10, writer=None, epoch=0,
                                  metric=None, calc_cmc=True, use_all=use_all, test=True, final=name_test)

            return

        if not os.path.exists(path_to_retmodel + '/tensorboard'):
            os.makedirs(path_to_retmodel + '/tensorboard')
        # initialize tensorboard writing
        writer = SummaryWriter(path_to_retmodel + '/tensorboard')

        self.model.cuda()
        self.model.train()

        criterion = nn.MSELoss(size_average=False).cuda()

        param_groups = self.model.parameters()

        lr = 0.000002

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



        evaluator.evaluate_cm(val_loader_int, val_loader_int_orig_, dataset.val_probe, dataset.val_gallery,
                              dataset_orig_.val_probe, dataset_orig_.val_gallery, 10, writer=writer, epoch=0,
                              metric=None, calc_cmc=True, use_all=use_all)

        evaluator.evaluate_validationloss(val_loader_ret, val_loader_int, criterion, 0, dataset.gallery, dataset.query,
                                       writer=writer)


        for epoch in range(start_epoch, epochs):
            adjust_lr(epoch, lr)
            print('Training epoch: ' + str(epoch))
            trainer.train(epoch, train_loader, optimizer, print_freq=50, writer=writer)


            top1 = evaluator.evaluate_cm(val_loader_int, val_loader_int_orig_, dataset.val_probe, dataset.val_gallery,
                                  dataset_orig_.val_probe, dataset_orig_.val_gallery, 10, writer=writer, epoch=epoch+1,
                                  metric=None, calc_cmc=True, use_all=use_all)

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

        model_load = models.create(self.model_arch, num_features=self.num_features,
                                   dropout=self.dropout, num_classes=self.num_classes_load)
        checkpoint = load_checkpoint(osp.join(path_to_retmodel, 'model_best.pth.tar'))
        model_load.load_state_dict(checkpoint['state_dict'])

	if self.num_classes_load != self.num_classes_infer:
	    model_load.num_classes = 0 

        print(checkpoint['epoch'])
        evaluator = Evaluator(model_load, self.model_orig)

        print("Modality 1: " + val_loader_int.dataset.root)
        print("Modality 2: " + val_loader_int_orig_.dataset.root)

        evaluator.evaluate_cm(val_loader_int, val_loader_int_orig_, dataset.val_probe, dataset.val_gallery,
                              dataset_orig_.val_probe, dataset_orig_.val_gallery, 10, writer=None, epoch=0,
                              metric=None, calc_cmc=True, use_all=use_all, final=name_val)
        print("Test:")
        evaluator.evaluate_cm(test_loader, test_loader_orig_, dataset.query, dataset.gallery,
                              dataset_orig_.query, dataset_orig_.gallery, 10, writer=None, epoch=0,
                              metric=None, calc_cmc=True, use_all=use_all, final=name_test)


def get_data(name, split_id, data_dir, height, width, batch_size, workers, combine_trainval, path_to_gt):

    root = data_dir +'/'+ name

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

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

    test_loader_ret = DataLoader(
        PreprocessorRetrain(test_set,
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return dataset, train_loader, val_loader_ret, val_loader_int, test_loader, test_loader_ret




