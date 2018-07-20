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
from sklearn.metrics import average_precision_score



class PostTrainer:
    def __init__(self, path_to_model):
        self.path_to_model = path_to_model


    def train (self, epochs, train_loader, val_loader, val_probe_loader, val_gallery_loader, test_probe_loader,
               test_gallery_loader,  writer=None):

        torch.set_num_threads(1)

        postnet = PostNet()
        # postnet.cuda()

        variance = 0.05
        for m in postnet.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, variance)
                pass

        lr = 0.0002

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, postnet.parameters()), lr=lr,
                                    momentum=0.9,
                                    weight_decay=0.0005,
                                    nesterov=True)
        best_top1 = float('inf')

        criterion = nn.MSELoss()#.cuda()


        # st2 = time.time()
        for e in range(epochs):

            postnet.eval()

            loss_val = 0
            for i, batch in enumerate(val_loader):
                b1 = batch[0]#.cuda()
                b2 = batch[1]#.cuda()
                gt = batch[2]#.cuda()
                gt = gt.type(torch.FloatTensor)#(torch.cuda.FloatTensor)

                outputs = postnet(b1, b2)
                outputs = torch.squeeze(outputs)
                loss = criterion(outputs, gt)
                loss_val += loss
            loss_val = loss_val/i
            print("Validation Loss: " + str(loss_val))
            if writer is not None:
                writer.add_scalar('Validation Loss', loss_val, e)

            is_best = loss_val < best_top1
            best_top1 = min(loss_val, best_top1)
            if is_best:
                print("Model is saved.")
                save_checkpoint({
                    'state_dict': postnet.state_dict(),
                    'epoch': e,
                    'best_top1': best_top1,
                }, is_best, fpath=osp.join(self.path_to_model, 'model_best.pth.tar'))

            postnet.train()

            loss_train = 0
            for i, batch in enumerate(train_loader):
                # print('Load ' + str(time.time()-st2))
                # st2 = time.time()
                b1 = batch[0]#.cuda()
                b2 = batch[1]#.cuda()
                gt = batch[2]#.cuda()
                gt = gt.type(torch.FloatTensor)#torch.cuda.FloatTensor)

                # st = time.time()
                outputs = postnet(b1, b2)
                # print('Net' + str(time.time()-st))

                outputs = torch.squeeze(outputs)
                loss = criterion(outputs, gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_train += loss

            loss_train = loss_train/i
            print("Loss Train: " + str(loss_train))
            if writer is not None:
                writer.add_scalar('Train Loss', loss_train, e)

            evaluate_cm(postnet, val_probe_loader, val_gallery_loader, writer, e, "Val ")
            evaluate_cm(postnet, test_probe_loader, test_gallery_loader, writer, e, "Test ")



def evaluate_cm(net, probe_loader, gallery_loader, writer, e, save_name):

    # query = [int(i[0].split("_")[1]) for i in probe_loader.dataset.dataset]
    top1 = 0
    top5 = 0
    top10 = 0

    torch.set_num_threads(1)

    for i in range(5):
        query_imgs, gal_imgs= \
            get_rand_images(probe_loader.dataset.dataset, gallery_loader.dataset.dataset)  # , imgs_from_q, imgs_from_g)

        query = [probe_loader.dataset.dataset[i][1] for i in query_imgs]
        gallery = torch.Tensor([gallery_loader.dataset.dataset[i][1] for i in gal_imgs])#.cuda()

        dist = torch.Tensor()#.cuda()

        net.eval()
        for j in range(len(query_imgs)):
            query_ = torch.Tensor(([query[j]]*len(gallery)))#.cuda()

            out = net(query_, gallery)

            dist = torch.cat((dist, out), dim=1)
            if j == 0 and i == 0:
                print(out[0:10])
            if out[j] == out[1] and j != 1:
                raise Exception('Badly initialized!')

        dist = dist.transpose(dim0=0, dim1=1)
        dist = dist.cpu().data.numpy()
        query_p = gallery_p = [i for i in range(len(query_imgs))]

        all = torch.empty(len(dist), 0)
        all = [sum(dist[i][i] <= dist[i]) for i in range(len(dist))]
        all = np.array(all)
        top1 += sum(all < 2) / len(all)
        top5 += sum(all < 6) / len(all)
        top10 += sum(all < 11) / len(all)

    top1_ = top1/5
    top5_ = top5/5
    top10_ = top10/5


    print(save_name + " Top1 " + str(top1_))
    print(save_name + " Top5 " + str(top5_))
    print(save_name + " Top10 " + str(top10_))

    if writer is not None:
        writer.add_scalar(save_name + 'Top1: ', top1_, e)
        writer.add_scalar(save_name + 'Top5: ', top5_, e)
        writer.add_scalar(save_name + 'Top10: ', top10_, e)


    #
    # cmc, mAP = compute_accuracy(dist, query_p, gallery_p)

    pass


def get_rand_images(query, gallery):
    # choose randomly from gallery and query

    query_p = [int(i[0].split("_")[0]) for i in query]
    gallery_p = [int(i[0].split("_")[0]) for i in gallery]


    liste = np.asarray(query_p)
    random_querys = [random.choice(np.where(liste == i)[0]) for i in np.unique(liste)]

    liste = np.asarray(gallery_p)
    random_gallery = [random.choice(np.where(liste == i)[0]) for i in np.unique(liste)]

    # query_imgs = [query_im[i] for i in random_querys]
    # gal_imgs = [gal_im[i] for i in random_gallery]

    return random_querys, random_gallery


def compute_accuracy(distmat, query_ids, gallery_ids, topk=10):
    single_gallery_shot = False
    first_match_break = True
    separate_camera_set = False
    m, n = distmat.shape
    # Fill up default values
    query_cams = np.zeros(m).astype(np.int32)
    gallery_cams = 2 * np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

    # Compute AP for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) | (gallery_cams[indices[i]] != query_cams[i]))
        if not np.any(matches[i, valid]): continue
        # Compute mAP
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]]
        # [valid]
        aps.append(average_precision_score(y_true, y_score))

        # Compute CMC
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])

        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1

    mAP = np.mean(aps)
    cmc = ret.cumsum() / num_valid_queries
    return cmc, mAP