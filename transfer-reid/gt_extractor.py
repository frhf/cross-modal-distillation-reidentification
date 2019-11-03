from __future__ import print_function, absolute_import

import torch
from torchvision import transforms
from torch.autograd import Variable

from PIL import Image
import sys
import glob
import pickle

sys.path.append('../open-reid/reid')
sys.path.append('../open-reid/reid/utils')

import models
import datasets
from utils.data.preprocessor import Preprocessor
from torch.utils.data import DataLoader
from utils.data import transforms as T
from collections import OrderedDict
import numpy as np
import math


class GtExtractor:
    def __init__(self, path_to_model):
        self.path_to_model = path_to_model
        self.load_model()


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
        self.model = models.create(self.model_arch, num_features=self.num_features,
                          dropout=0, num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint['state_dict'])

        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        if self.model_arch == 'inception':
            height, width = (144, 56)
        else:
            height, width = (256, 128)

        self.test_transformer = T.Compose([
            T.RectScale(height, width),
            T.ToTensor(),
            normalizer,
        ])

    def extract_gt(self, name, data_dir='/export/livia/data/FHafner/data/',
                   path_to_save_gt=None, extract_for='train', save_as=None, batch_size=64, workers=2):

        if save_as is None:
            save_as = extract_for

        root = data_dir + '/' + name

        print('Extracting GT from ' + extract_for + ' for ' + name)

        dataset = datasets.create(name, root, split_id=0)
        if extract_for == 'train':
            dataset_eval = dataset.train
        elif extract_for == 'val_probe':
            dataset_eval = dataset.val_probe
        elif extract_for == 'val_gallery':
            dataset_eval = dataset.val_gallery
        elif extract_for == 'query':
            dataset_eval = dataset.query
        elif extract_for == 'gallery':
            dataset_eval = dataset.gallery
        else:
            raise RuntimeError("Please choose extraction from 'train', 'val_query', 'val_galler', 'query' and 'gallery'")

        loader = DataLoader(
            Preprocessor(dataset_eval,
                         root=dataset.images_dir, transform=self.test_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=False, pin_memory=False)

        with torch.no_grad():
            self.model.eval()
            self.model.cuda()
            torch.set_num_threads(1)

            gt_eval = []
            for i, inputs in enumerate(loader):
                names = inputs[1]
                inputs = inputs[0].cuda()
                outputs = self.model(inputs)
                outputs = outputs.cpu().detach().numpy()
                gt_eval += [[name, output] for name, output in zip(names, outputs)]


            with open(path_to_save_gt + '/gt_' + save_as + '.txt', 'wb') as f:
                pickle.dump(gt_eval, f)

            return

    def extract_gt_av(self, name, name_transfer, data_dir='/export/livia/data/FHafner/data/',
                   path_to_save_gt=None, extract_for='train', batch_size=64, workers=2):

        root = data_dir + '/' + name
        dataset = datasets.create(name, root, split_id=0)

        if extract_for == 'train':
            dataset_eval = dataset.train
        elif extract_for == 'val':
            dataset_eval = dataset.val_probe + dataset.val_gallery
            dataset_eval = list(OrderedDict.fromkeys(dataset_eval))
        elif extract_for == 'query':
            dataset_eval = dataset.query + dataset.gallery
            dataset_eval = list(OrderedDict.fromkeys(dataset_eval))



        loader = DataLoader(
            Preprocessor(dataset_eval,
                         root=dataset.images_dir, transform=self.test_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=False, pin_memory=False)

        with torch.no_grad():
            self.model.eval()
            self.model.cuda()
            torch.set_num_threads(1)

            gt_eval = []
            for i, inputs in enumerate(loader):
                if i%20 == 0:
                    print(i)
                person = inputs[2]
                inputs = inputs[0].cuda()
                outputs = self.model(inputs)
                outputs = outputs.cpu().detach().numpy()
                gt_eval += [[person, output] for person, output in zip(person, outputs)]

        dict_ ={}
        for idxs in dataset.split[extract_for]:
            # attention idxs in
            if name == 'tum' or name == 'tum_depth':
                idxs -= 1

            idx = [int(ex[0]) == idxs for ex in gt_eval]
            tensrs = np.array([part[1] for i, part in enumerate(gt_eval) if idx[i] == True])
            mean_ = np.mean(tensrs, axis=0)
            # print (idxs)
            # if math.isnan(mean_):
            #     pass

            dict_[idxs] = mean_
            mean_ = None
            tensrs = None

        root = data_dir + '/' + name_transfer
        dataset = datasets.create(name_transfer, root, split_id=0)

        if extract_for == 'train':
            new_gt = [[gt[0], dict_[gt[1]]] for gt in dataset.train]
            with open(path_to_save_gt + '/gt_' + extract_for + '.txt', 'wb') as f:
                pickle.dump(new_gt, f)

        elif extract_for == 'val':
            gt_val_probe = [[gt[0], dict_[gt[1]]] for gt in dataset.val_probe]
            gt_val_gallery = [[gt[0], dict_[gt[1]]] for gt in dataset.val_gallery]

            with open(path_to_save_gt + '/gt_val_probe.txt', 'wb') as f:
                pickle.dump(gt_val_probe, f)

            with open(path_to_save_gt + '/gt_val_gallery.txt', 'wb') as f:
                pickle.dump(gt_val_gallery, f)

        elif extract_for == 'query':
            gt_query = [[gt[0], dict_[gt[1]]] for gt in dataset.query]
            gt_gallery = [[gt[0], dict_[gt[1]]] for gt in dataset.gallery]

            with open(path_to_save_gt + '/gt_query.txt', 'wb') as f:
                pickle.dump(gt_query, f)

            with open(path_to_save_gt + '/gt_gallery.txt', 'wb') as f:
                pickle.dump(gt_gallery, f)

        pass
        

