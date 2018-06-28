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


class Retrainer:
    def __init__(self, path_to_model, dropout=0):
        self.path_to_model = path_to_model
        self.dropout = dropout
        self.model = self.load_model()
        self.model_orig = copy.deepcopy(self.model)
        self.model_orig.eval()

    def load_model(self):
        name_dict = {
            110: 'resnet18',
            190: 'resnet34',
            203: 'inception',
            275: 'resnet50',
        }

        checkpoint = torch.load(self.path_to_model)
        self.model_arch = name_dict[len(checkpoint['state_dict'])]
        self.num_features = checkpoint['state_dict']['classifier.weight'].shape[1]
        self.num_classes = checkpoint['state_dict']['classifier.weight'].shape[0]
        model = models.create(self.model_arch, num_features=self.num_features,
                          dropout=self.dropout, num_classes=self.num_classes)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        return model

    def retrain(self, dataset_ret, dataset_orig, path_to_gt, root='/export/livia/data/FHafner/data/',
                epochs=50, split_id=0, start_epoch=0, batch_size=64, workers=2,
                combine_trainval=False, path_to_retmodel=None):

        if self.model_arch == 'inception':
            height, width = (144, 56)
        else:
            height, width = (256, 128)

        dataset, train_loader, val_loader_ret, val_loader_int, test_loader, test_loader_ret = \
            get_data(dataset_ret, split_id, root, height, width, batch_size, workers, combine_trainval, path_to_gt)

        if not os.path.exists(path_to_retmodel + 'tensorboard'):
            os.makedirs(path_to_retmodel + 'tensorboard')
        # initialize tensorboard writing
        writer = SummaryWriter(path_to_retmodel + 'tensorboard')

        self.model.cuda()
        self.model.train()

        criterion = nn.MSELoss(size_average=False).cuda()
        #criterion = nn.L1Loss(size_average=False).cuda()

        param_groups = self.model.parameters()

        # optimizer = torch.optim.Adam(params=param_groups, weight_decay=1)

        lr = 0.00002
        optimizer = torch.optim.SGD(param_groups, lr=lr,
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

        for epoch in range(start_epoch, epochs):
            adjust_lr(epoch, lr)

            # evaluate in retrained domain
            top1 = evaluator.evaluate_single_shot(dataset.query, dataset.gallery, 1, writer,
                                                       epoch, root + dataset_ret, height, width, name2save="Reid in Retrain Domain: ")

            evaluator.evaluate_single_shot(dataset.train, dataset.train, 1, writer,
                                                       epoch, root + dataset_ret, height, width, name2save="Training data: ")

            evaluator.evaluate_one(dataset.train, dataset.train, 1, writer,
                                                       epoch, root + dataset_ret, height, width, root2=root + dataset_orig)
            # evaluate CM
            evaluator.evaluate_single_shot_cm(dataset.query, dataset.gallery, 1, writer,
                                                       epoch, root + dataset_ret, height, width, root + dataset_orig, 'Cross_modal: ')

            # if epoch % 4 == 0:
            evaluator.evaluate_retrain(test_loader_ret, val_loader_int, criterion, epoch, dataset.gallery, dataset.query,
                                       writer=writer)

            trainer.train(epoch, train_loader, optimizer, print_freq=50, writer=writer)

            # save model if its best
            is_best = top1 > best_top1
            if is_best:
                print("Model is saved.")
                save_checkpoint({
                    'state_dict': self.model.state_dict(),
                    'epoch': epoch + 1,
                    'best_top1': best_top1,
                }, is_best, fpath=osp.join(path_to_retmodel, 'model_best.pth.tar'))


def get_data(name, split_id, data_dir, height, width, batch_size, workers, combine_trainval, path_to_gt):

    root = data_dir +'/'+ name

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])


    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        # T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_set = pickle.load(open(path_to_gt + 'gt_train.txt', "rb"))
    print('average loaded.')
    val_set = pickle.load(open(path_to_gt + 'gt_val.txt', "rb"))

    query_set = pickle.load(open(path_to_gt + 'gt_query.txt', "rb"))
    gallery_set = pickle.load(open(path_to_gt + 'gt_gallery.txt', "rb"))


    if combine_trainval:
        train_s = train_set + val_set
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
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    test_loader_ret = DataLoader(
        PreprocessorRetrain(query_set + gallery_set,
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return dataset, train_loader, val_loader_ret, val_loader_int, test_loader, test_loader_ret
