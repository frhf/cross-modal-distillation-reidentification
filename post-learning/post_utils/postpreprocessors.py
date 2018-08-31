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
import random
from post_model import PostNet
import numpy as np
import time
from utils.serialization import load_checkpoint, save_checkpoint




class PreprocessorPost(object):
    def __init__(self, path_to_gt1, path_to_gt2):
        super(PreprocessorPost, self).__init__()
        self.path_to_gt1 = path_to_gt1
        self.path_to_gt2 = path_to_gt2

        self.dataset1 = pickle.load(open(path_to_gt1, "rb"))
        # self.arr1 = [torch.Tensor(i[1]).cuda() for i in self.dataset1]


        self.dataset2 = pickle.load(open(path_to_gt2, "rb"))
        # self.arr2 = [torch.Tensor(i[1]).cuda() for i in self.dataset2]


    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):

        fname1, enc1 = self.dataset1[index]
        # enc1 = self.arr1[index]

        pers1, _, _ = fname1.split("_")
        pers1 = int(pers1)

        if bool(random.getrandbits(1)):
            index2 = random.randint(0, len(self.dataset2)-1)
            fname2, enc2 = self.dataset2[index2]
            # enc2 = self.arr2[index2]
            pers2, _, _ = fname2.split("_")
            pers2 = int(pers2)
        # get example of same person
        else:
            rel = np.where([int(ex[0].split("_")[0]) == pers1 for ex in self.dataset2])
            rand_ = random.choice(rel[0])
            fname2, enc2 = self.dataset2[rand_]
            # enc2 = self.arr2[rand_]
            pers2, _, _ = fname2.split("_")
            pers2 = int(pers2)


        if pers1 == pers2:
            same = 1
        else:
            same = 0

        return enc1, enc2, same

class PreprocessorPostEval(object):
    def __init__(self, path_to_gt):
        super(PreprocessorPostEval, self).__init__()
        self.path_to_gt1 = path_to_gt

        self.dataset = pickle.load(open(path_to_gt, "rb"))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):

        fname1, enc1 = self.dataset1[index]
        pers1, _, _ = fname1.split("_")
        pers1 = int(pers1)

        return fname1, enc, pers1