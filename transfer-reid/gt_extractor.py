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
#from utils.data.preprocessor import Preprocessor
# from torch.utils.data import DataLoader
from utils.data import transforms as T



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

    def extract_gt(self, name, data_dir='/export/livia/data/FHafner/data/',
                   path_to_save_gt=None):

        with torch.no_grad():

            torch.set_num_threads(1)
            root = data_dir + '/' + name

            if self.model_arch == 'inception':
                height, width = (256, 128)
            else:
                height, width = (144, 56)

            dataset = datasets.create(name, root, split_id=0)
            dataset_train = [idx for idx, _, _ in dataset.train]
            dataset_val = [idx for idx, _, _ in dataset.val]
            dataset_query = [idx for idx, _, _ in dataset.query]
            dataset_gallery = [idx for idx, _, _ in dataset.gallery]

            self.model = self.model.cuda()
            self.model.eval()

            # save groundtruth of training data
            # gt_train = []
            # for idx, img_path in enumerate(dataset_train):
            #     img = image_loader(root + '/images/' + img_path, height, width)
            #     out = self.model(img)
            #     out = out.cpu().detach().numpy()[0]
            #     gt_train.append([img_path, out])
            #
            #     if idx % 50 == 0:
            #        print(img_path)
            #
            # with open(path_to_save_gt + 'gt_train.txt', 'wb') as f:
            #     pickle.dump(gt_train, f)
            #
            # del gt_train

            # save groundtruth of validation data
            # gt_val = []
            # for idx, img_path in enumerate(dataset_val):
            #     img = image_loader(root + '/images/' + img_path, height, width)
            #
            #     # TODO: Include batch processing
            #     out = self.model(img)
            #     out = out.cpu().detach().numpy()[0]
            #     gt_val.append([img_path, out])
            #
            #     if idx % 200 == 0:
            #        print(img_path)
            #
            # with open(path_to_save_gt + 'gt_val.txt', 'wb') as f:
            #     pickle.dump(gt_val, f)
            #
            # del gt_val

            # save groundtruth of gallery data
            gt_gal = []
            for idx, img_path in enumerate(dataset_gallery):
                img = image_loader(root + '/images/' + img_path, height, width)

                # TODO: Include batch processing
                out = self.model(img)
                out = out.cpu().detach().numpy()[0]
                gt_gal.append([img_path, out])

                if idx % 200 == 0:
                    print(img_path)

            with open(path_to_save_gt + 'gt_gal.txt', 'wb') as f:
                pickle.dump(gt_gal, f)

            del gt_gal

            # save groundtruth of query data
            gt_query = []
            for idx, img_path in enumerate(dataset_query):
                img = image_loader(root + '/images/' + img_path, height, width)

                # TODO: Include batch processing
                out = self.model(img)
                out = out.cpu().detach().numpy()[0]
                gt_query.append([img_path, out])

                if idx % 200 == 0:
                    print(img_path)

            with open(path_to_save_gt + 'gt_query.txt', 'wb') as f:
                pickle.dump(gt_query, f)

            del gt_query

            return


def image_loader(image_name, height, width):
    """load image, returns cuda tensor"""
    torch.set_num_threads(1)

    # initialize loading and normalization
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    img = Image.open(image_name).convert('RGB')
    img_t = test_transformer(img).cuda()
    img_t = img_t.unsqueeze(0)

    return img_t
    # loader = transforms.Compose([transforms.ToTensor()])
    # image = Image.open(image_name)
    # image = loader(image).float()
    # image = Variable(image, requires_grad=True)
    # image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    # return image.cuda()