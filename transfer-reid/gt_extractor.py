from __future__ import print_function, absolute_import

import torch
from torchvision import transforms
from torch.autograd import Variable

from PIL import Image
import sys
import glob
import pickle

sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid')
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/utils')

import models
import datasets

class GtExtractor:
    def __init__(self, path_to_model):
        self.path_to_model = path_to_model
        self.load_model()

    def load_model(self):
        name_dict = {
            110: 'resnet18',
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

    def extract_gt(self, name, data_dir='/export/livia/home/vision/FHafner/masterthesis/open-reid/examples/data',
                   path_to_save_gt=None):

        root = data_dir + '/' + name

        dataset = datasets.create(name, root, split_id=0)
        dataset_train = [idx for idx, _, _ in dataset.train]
        dataset_val = [idx for idx, _, _ in dataset.val]

        self.model = self.model.cuda()
        self.model.eval()

        # save groundtruth of training data
        gt_train = []
        for idx, img_path in enumerate(dataset_train):
            img = image_loader(root + '/images/' + img_path)
            out = self.model(img)
            out = out.cpu().detach().numpy()[0]
            gt_train.append([img_path, out])

            if idx % 50 == 0:
               print(img_path)

        with open(path_to_save_gt + 'gt_train.txt', 'wb') as f:
            pickle.dump(gt_train, f)

        gt_val = []
        # save groundtruth of validation data
        for idx, img_path in enumerate(dataset_val):
            img = image_loader(root + '/images/' + img_path)

            # TODO: Include batch processing
            out = self.model(img)
            out = out.cpu().detach().numpy()[0]
            gt_val.append([img_path, out])

            if idx % 50 == 0:
               print(img_path)

        with open(path_to_save_gt + 'gt_val.txt', 'wb') as f:
            pickle.dump(gt_val, f)

        return gt_train, gt_val


def image_loader(image_name):
    """load image, returns cuda tensor"""

    loader = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    return image.cuda()