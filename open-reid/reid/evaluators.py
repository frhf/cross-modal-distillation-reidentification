from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
import numpy as np
import sys
sys.path.append('.')

from torch.autograd import Variable

from evaluation_metrics import cmc, mean_ap
from feature_extraction import extract_cnn_feature
from utils.meters import AverageMeter
import random
from utils.data.preprocessor import Preprocessor
from utils.data import transforms as T
import os.path as osp
from PIL import Image
from sklearn.metrics import average_precision_score
import scipy.io as sio
from utils.visualization_tools import make_comparison_img, make_comparison_img_cm
import pickle

# evaluates NN and saves time for doing so
def extract_features(model, data_loader, print_freq=1, metric=None, n_batches=None):
    # evaluation mode for model
    torch.set_num_threads(1)
    model.cuda()
    model.eval()

    if n_batches is None:
        n_batches = len(data_loader)

    # time for dataprocessing
    batch_time = AverageMeter()
    # # updates for data loading
    data_time = AverageMeter()

    # dictionary which remembers which input after which was given
    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        # updates for data loading
        data_time.update(time.time() - end)
        imgs_c = Variable(imgs.cuda(), requires_grad=False)
        # evaluates model
        # outputs = extract_cnn_feature(model, imgs)
        outputs = model(imgs_c)
        outputs = outputs.data.cpu()
        imgs_c = None
        imgs = None
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        # time for dataprocessing
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % 100 == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

        if i == n_batches:
            break
    print(output.shape)



# calculates pairwise distances
def pairwise_distance(features1, query=None, gallery=None, metric=None, features2=None):
    if features2 is None:
        features2 = features1

    torch.set_num_threads(1)


    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features1[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features2[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    # this is a2-2ab+b2
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), writer=None,
                 epoch=None, calc_cmc=True, use_all=True,
                 save_as="", same=None, final=[]):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    beg = time.time()
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams, use_all=use_all, same=same)
    print(save_as + 'Mean AP: {:4.1%}'.format(mAP))

    if calc_cmc:
        # Compute all kinds of CMC scores
        cmc_configs = {
            'cuhk03': dict(separate_camera_set=True,
                           single_gallery_shot=True,
                           first_match_break=False,
                           use_all=use_all,
                           same=same),
        }
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                query_cams, gallery_cams, **params)
                      for name, params in cmc_configs.items()}


        print(save_as)
        print('CMC Scores{:>12}'
             .format('cuhk03')) 
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'
                  .format(k, cmc_scores['cuhk03'][k - 1])),
                      
    if writer is not None:
        writer.add_scalar(save_as + ' Rank 1', cmc_scores['cuhk03'][0], epoch)
        writer.add_scalar(save_as + 'Rank 5', cmc_scores['cuhk03'][4], epoch)
        writer.add_scalar(save_as + 'Rank 10', cmc_scores['cuhk03'][9], epoch)
        writer.add_scalar(save_as + ' mAP', mAP, epoch)

    if not final == []:

        path = '/export/livia/data/FHafner/data/outputs_new/'
        file = open(path + final, 'ab+')
        pickle.dump([cmc_scores['cuhk03'][0], cmc_scores['cuhk03'][4], cmc_scores['cuhk03'][9], mAP], file)
        file.close()

    return mAP

class Evaluator(object):
    def __init__(self, model, model_cm=None):
        super(Evaluator, self).__init__()
        self.model = model
        if model_cm is not None:
            self.model_cm = model_cm
        # makes cross-modal evaluation within one modal possible
        else:
            self.model_cm = model

    def evaluate(self, data_loader, query, gallery, print_freq, writer=None, epoch=None, metric=None, calc_cmc=False,
                 use_all=False, final=[]):
        torch.set_num_threads(1)
        features, _ = extract_features(self.model, data_loader, print_freq)

	# Exception for specific datasets
        if data_loader.dataset.root == '/export/livia/data/FHafner/data/synthia_depth/images' or \
                data_loader.dataset.root == '/export/livia/data/FHafner/data/synthia/images' or \
                data_loader.dataset.root == '/export/livia/data/FHafner/data/iit_depth/images' or \
                data_loader.dataset.root == '/export/livia/data/FHafner/data/iit/images':
            query_ad = query
            gallery_ad = gallery
        else:
            query_ad, gallery_ad = get_rand(query, gallery)

	# If all images in gallery and query are supposed to be used, avoid that exactly same image is evaluated.
        if use_all:
            same = getsame(query_ad, gallery_ad)
        else:
            same = None

        distmat = pairwise_distance(features, query_ad, gallery_ad, metric=metric)
        return evaluate_all(distmat, query=query_ad, gallery=gallery_ad, writer=writer, epoch=epoch, calc_cmc=calc_cmc,
                            use_all=use_all, same=same, final=final)

    # Use for evaluation of cross-modal re-identification
    def evaluate_cm(self, data_loader1, data_loader2, query1, gallery1, query2, gallery2, print_freq, writer=None,
                    epoch=None, metric=None, calc_cmc=False, use_all=False, test=False, final=[]):
        torch.set_num_threads(1)
        features1, _ = extract_features(self.model, data_loader1, print_freq)
        features2, _ = extract_features(self.model_cm, data_loader2, print_freq)


	# Exception for Synthia and Sysu.
	# Else find same 
        if data_loader1.dataset.root.split('/')[-2] == 'synthia_depth' or \
                data_loader1.dataset.root.split('/')[-2] == 'synthia':
            query1_r = query1
            gallery1_r = gallery1
            query2_r = query2
            gallery2_r = gallery2
        elif data_loader1.dataset.root.split('/')[-2] == 'sysu' or \
                data_loader1.dataset.root.split('/')[-2] == 'sysu_ir':
            query1_r, gallery1_r = get_rand(query1, gallery1, query_am=20, gal_am=10)
            query2_r, gallery2_r = get_rand(query2, gallery2, query_am=20, gal_am=10)
        else:
            query1_r, gallery1_r = get_rand(query1, gallery1, query_am=50, gal_am=50)
            query2_r, gallery2_r = get_rand(query2, gallery2, query_am=50, gal_am=50)

        # one modal 1
        # ATTENTION SYSUIR TEST
        if data_loader1.dataset.root.split("/")[-2]=='sysu_ir' and not test or use_all:
            use_all_temp = True
            same = getsame(query2_r, gallery2_r)

        else:
            use_all_temp = False
            same= None


        # one modal 1
        distmat1 = pairwise_distance(features1, query1_r, gallery1_r, metric=metric)
        evaluate_all(distmat1, query=query1_r, gallery=gallery1_r, writer=writer, epoch=epoch, calc_cmc=calc_cmc,
                            use_all=use_all_temp, save_as='Modality 1 ', same=same, final=final)

        # one modal 2
        if data_loader2.dataset.root.split("/")[-2]=='sysu_ir' and not test or use_all:
            use_all_temp = True
            same = getsame(query2_r, gallery2_r)

        else:
            use_all_temp = False
            same= None

        distmat2 = pairwise_distance(features2, query2_r, gallery2_r, metric=metric)
        evaluate_all(distmat2, query=query2_r, gallery=gallery2_r, writer=writer, epoch=epoch, calc_cmc=calc_cmc,
                            use_all=use_all_temp, save_as='Modality 2 ', same=same, final=final)

        if use_all:
            same = getsame(query1_r, gallery2_r)

        # cross modal 1
        distmat_cm1 = pairwise_distance(features1, query1_r, gallery2_r, metric=metric, features2=features2)
        cm1 = evaluate_all(distmat_cm1, query=query1_r, gallery=gallery2_r, writer=writer, epoch=epoch, calc_cmc=calc_cmc,
                            use_all=use_all, save_as='Cross-Modality from 1 to 2 ', same=same, final=final)

        if use_all:
            same = getsame(query2_r, gallery1_r)

        # cross modal 2
        distmat_cm2 = pairwise_distance(features2, query2_r, gallery1_r, metric=metric, features2=features1)
        cm2 = evaluate_all(distmat_cm2, query=query2_r, gallery=gallery1_r, writer=writer, epoch=epoch, calc_cmc=calc_cmc,
                            use_all=use_all, save_as='Cross-Modality from 2 to 1 ', same=same, final=final)

        return cm1 + cm2


    # function which gives visualization for cross modal task
    def make_comp_cm(self, data_loader1, data_loader2, query1, gallery1, query2, gallery2, print_freq, writer=None, epoch=None,
                  metric=None, calc_cmc=False,
                  use_all=False):
        if data_loader2 is None:
            data_loader2 = data_loader1
        torch.set_num_threads(1)
        features1, _ = extract_features(self.model, data_loader1, print_freq)
        features2, _ = extract_features(self.model_cm, data_loader2, print_freq)


        query_ad, gallery_ad = get_rand(query1, gallery2, 1 ,1)
        distmat_cm1 = pairwise_distance(features1, query_ad, gallery_ad, metric=metric, features2=features2)


        if use_all:
            same = getsame(query_ad, gallery_ad)
        else:
            same = None

        make_comparison_img_cm(data_loader1.dataset.root, data_loader2.dataset.root, distmat_cm1, query_ad, gallery_ad)


    # function which gives visualization for single modal task
    def make_comp(self, data_loader1, query, gallery, print_freq, writer=None, epoch=None, metric=None, calc_cmc=False,
                 use_all=False, data_loader2=None):
        if data_loader2 is None:
            data_loader2 = data_loader1
        torch.set_num_threads(1)
        features, _ = extract_features(self.model, data_loader1, print_freq)


        query_ad, gallery_ad = get_rand(query, gallery, 1 ,1)

        if use_all:
            same = getsame(query_ad, gallery_ad)
        else:
            same = None

        distmat = pairwise_distance(features, query_ad, gallery_ad, metric=metric)
        make_comparison_img(data_loader1.dataset.root, data_loader2.dataset.root, distmat, query_ad, gallery_ad)


    # evaluates validation loss
    def evaluate_validationloss(self, val_loader_ret, val_loader_int, criterion, epoch, query, gallery, writer=None):

        with torch.no_grad():
            torch.set_num_threads(1)
            self.model.eval()
            overall_loss = 0
            for i, batch in enumerate(val_loader_ret):
                imgs = Variable(batch[0].cuda(), requires_grad=False)
                out = self.model(imgs)
                target = Variable(batch[2].cuda(), requires_grad=False)
                loss = Variable(criterion(out, target), requires_grad=False)
                loss = torch.sqrt(loss).sum()/len(imgs)
                overall_loss += loss

                # free memory
                imgs = None
                out = None
                target = None

            # get average euclidean distance
            overall_loss = overall_loss/i
            overall_loss_n = overall_loss.cpu().numpy()
            print('Validation Loss: ' + str(overall_loss_n))

            if writer is not None:
                writer.add_scalar('ValidationLoss', overall_loss_n, epoch)

            return overall_loss_n

    # function which is only for SYSU dataset. Gives output according to evaluation scheme
    def evaluate_all_and_save_sysu(self, query, gallery, save_to, height=None, width=None):

        with torch.no_grad():

            # initialize loading and normalization
            normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

            test_transformer = T.Compose([
                T.RectScale(height, width),
                T.ToTensor(),
                normalizer,
            ])

            # load gallery
            imgs_batch = Variable().cuda()
            # out_all = np.empty(0)
            array_ = np.empty([1, 6, 533, 50, self.model.num_classes])
            self.model.cuda()
            self.model.eval()

            for i, inputs in enumerate(query):
                imgs, a, _, _ = inputs
                inputs = [Variable(imgs).cuda()]
                outputs = self.model(*inputs)


                liste = ([i.split('_') for i in a])
                pers = [int(perso[0]) for perso in liste]
                cam = [int(camo[1]) for camo in liste]
                num = [int(numo[2].split('.')[0]) for numo in liste]
                for i in range(len(outputs)):
                    array_[0, cam[i] - 1, pers[i], num[i] - 1] = outputs[i]
                    # out_all = np.concatenate([out_all, np.array(outputs)])

            self.model_cm.cuda()
            self.model_cm.eval()
            for i, inputs in enumerate(gallery):
                imgs, a, _, _ = inputs
                inputs = [Variable(imgs).cuda()]
                outputs = self.model_cm(*inputs)


                liste = ([i.split('_') for i in a])
                pers = [int(perso[0]) for perso in liste]
                cam = [int(camo[1]) for camo in liste]
                num = [int(numo[2].split('.')[0]) for numo in liste]
                for i in range(len(outputs)):
                    array_[0, cam[i] - 1, pers[i], num[i] - 1] = outputs[i]
                    # out_all = np.concatenate([out_all, np.array(outputs)])

            sio.savemat(save_to + '/np_vector.mat', {'vect': array_})


# get random images from query and gallery
def get_rand_images(query, gallery):
    # choose randomly from gallery and query
    liste = np.asarray([q[1] for q in query])
    random_querys = [random.choice(np.where(liste == i)[0]) for i in np.unique(liste)]

    liste = np.asarray([q[1] for q in gallery])
    random_gallery = [random.choice(np.where(liste == i)[0]) for i in np.unique(liste)]

    gallery_ = np.asarray(gallery)
    query_ = np.asarray(query)

    gal_imgs = gallery_[random_gallery]
    query_imgs = query_[random_querys]

    return query_imgs, gal_imgs


# get random images from query and gallery
def get_rand(query, gallery, query_am=50, gal_am=50):
    nums = np.array([i[1] for i in gallery])
    vec = [np.where(nums == i)[0] for i in range(nums.max()) if np.where(nums == i)[0].size > 0]
    rand_ = np.sort(np.array([np.random.choice(x, gal_am) for x in vec]).flatten())
    gallery_ad = [gallery[rel] for rel in rand_]

    nums = np.array([i[1] for i in query])
    vec = [np.where(nums == i)[0] for i in range(nums.max()) if np.where(nums == i)[0].size > 0]
    rand_ = np.sort(np.array([np.random.choice(x, query_am) for x in vec]).flatten())
    query_ad = [query[rel] for rel in rand_]

    return query_ad, gallery_ad

# get binary mask if query and gallery images contain exactly the same images.
def getsame(query, gallery):

    same = np.ones((len(query), len(gallery)), dtype=bool)
    for i in range(len(query)):
        for j in range(len(gallery)):
            if query[i] == gallery[j]:
                same[i][j] = 0

    return same



