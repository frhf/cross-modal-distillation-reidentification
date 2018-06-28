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
from utils.data.preprocessor import Preprocessor
from torch.utils.data import DataLoader
from utils.data import transforms as T



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
                          dropout=0.2, num_classes=self.num_classes)
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
                   path_to_save_gt=None, extract_for='train', batch_size=64, workers=2):

        root = data_dir + '/' + name

        print('Extracting GT from ' + extract_for + ' for ' + name)

        dataset = datasets.create(name, root, split_id=0)
        if extract_for == 'train':
            dataset_eval = dataset.train
        elif extract_for == 'val':
            dataset_eval = dataset.val
        elif extract_for == 'query':
            dataset_eval = dataset.query
        elif extract_for == 'gallery':
            dataset_eval = dataset.gallery
        else:
            raise RuntimeError("Please choose extraction from 'train', 'val', 'query' and 'gallery'")

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


            with open(path_to_save_gt + '/gt_' + extract_for + '.txt', 'wb') as f:
                pickle.dump(gt_eval, f)

            # root = data_dir + '/' + name
            #
            # # if self.model_arch == 'inception':
            # #     height, width = (256, 128)
            # # else:
            # #     height, width = (144, 56)
            #
            #
            # self.model = self.model.cuda()
            # self.model.eval()
            #
            # # save groundtruth of query data
            # gt_query = []
            # for idx, img_path in enumerate(dataset_eval):
            #     img = image_loader(root + '/images/' + img_path, height, width)
            #
            #     # TODO: Include batch processing
            #     out = self.model(img)
            #     out = out.cpu().detach().numpy()[0]
            #     gt_query.append([img_path, out])
            #
            #     if idx % 200 == 0:
            #         print(img_path)
            #
            # with open(path_to_save_gt + 'gt_query.txt', 'wb') as f:
            #     pickle.dump(gt_query, f)
            #
            # del gt_query

            return


# def image_loader(image_name, height, width):
#     """load image, returns cuda tensor"""
#     torch.set_num_threads(1)
#
#     # initialize loading and normalization
#     normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#
#     test_transformer = T.Compose([
#         T.RectScale(height, width),
#         T.ToTensor(),
#         normalizer,
#     ])
#
#     img = Image.open(image_name).convert('RGB')
#     img_t = test_transformer(img).cuda()
#     img_t = img_t.unsqueeze(0)
#
#     return img_t
    # loader = transforms.Compose([transforms.ToTensor()])
    # image = Image.open(image_name)
    # image = loader(image).float()
    # image = Variable(image, requires_grad=True)
    # image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    # return image.cuda()