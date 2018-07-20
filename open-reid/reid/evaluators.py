from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
import numpy as np
import sys
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/')
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/utils')

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



# evaluates NN and saves time for doing so
def extract_features(model, data_loader, print_freq=1, metric=None, n_batches=None):
    # evaluation mode for model
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

    return features, labels


# calculates pairwise distances
def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
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
                 cmc_topk=(1, 5, 10), writer=None, epoch=None, calc_cmc=False, use_all=False):
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
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams, use_all=use_all)
    print('Mean AP: {:4.1%}'.format(mAP))
    # print("mAP: " + str(time.time() - beg))

    if calc_cmc:
        # Compute all kinds of CMC scores
        cmc_configs = {
            # 'allshots': dict(separate_camera_set=False,
            #                  single_gallery_shot=False,
            #                  first_match_break=False),
            'cuhk03': dict(separate_camera_set=True,
                           single_gallery_shot=True,
                           first_match_break=False,
                           use_all = use_all),
            # 'market1501': dict(separate_camera_set=False,
            #                    single_gallery_shot=False,
            #                    first_match_break=True)}
        }
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                query_cams, gallery_cams, **params)
                      for name, params in cmc_configs.items()}

        #print('CMC Scores{:>12}{:>12}{:>12}'
         #     .format('allshots', 'cuhk03', 'market1501'))
        print('CMC Scores{:>12}'#{:>12}{:>12}'
             .format('cuhk03')) #'allshots', 'cuhk03', 'market1501'))
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'#{:12.1%}{:12.1%}'
                  .format(k, #cmc_scores['allshots'][k - 1],
                          cmc_scores['cuhk03'][k - 1]))#,
                          #cmc_scores['market1501'][k - 1]))
    #
    if writer is not None:
        writer.add_scalar('Rank 1', cmc_scores['cuhk03'][0], epoch)
        writer.add_scalar('Rank 5', cmc_scores['cuhk03'][4], epoch)
        writer.add_scalar('Rank 10', cmc_scores['cuhk03'][9], epoch)
        writer.add_scalar('mAP', mAP, epoch)

    # Use the allshots cmc top-1 score for validation criterion
    # return cmc_scores['cuhk03'][0]
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
                 use_all=False):
        features, _ = extract_features(self.model, data_loader, print_freq)

        # query = [i for no, i in enumerate(query) if no % 20 == 0]
        # gallery = [i for no, i in enumerate(gallery) if no % 20 == 0]
        nums = np.array([i[1] for i in gallery])
        vec = [np.where(nums == i)[0] for i in range(nums.max()) if np.where(nums == i)[0].size > 0]
        rand_ = np.sort(np.array([np.random.choice(x, 10) for x in vec]).flatten())
        gallery_ad = [gallery[rel] for rel in rand_]

        nums = np.array([i[1] for i in query])
        vec = [np.where(nums == i)[0] for i in range(nums.max()) if np.where(nums == i)[0].size > 0]
        rand_ = np.sort(np.array([np.random.choice(x, 20) for x in vec]).flatten())
        query_ad = [query[rel] for rel in rand_]

        distmat = pairwise_distance(features, query_ad, gallery_ad, metric=metric)
        return evaluate_all(distmat, query=query_ad, gallery=gallery_ad, writer=writer, epoch=epoch, calc_cmc=calc_cmc,
                            use_all=use_all)

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


    # def evaluate_partly(self, data_loader, query, gallery, print_freq, writer=None, epoch=None, n_batches=None, metric=None):
    #     features, labels = extract_features(self.model, data_loader, print_freq, n_batches=n_batches)
    #     acc1 = cmc_partly(features, labels, writer, epoch)
    #     return acc1

    # evaluates single query, single gallery. Therefore, computationally cheap evaluation.
    def evaluate_single_shot(self, query, gallery, print_freq, writer=None, epoch=None, root=None, height=None,
                             width=None, name2save="", evaluations=5):

        # if len(query) != len(gallery):
        #     query = gallery

        with torch.no_grad():
            torch.set_num_threads(1)
            self.model.cuda()
            self.model.eval()

            top1_ = []
            top5_ = []
            top10_ = []
            mAP_= []

            # evaluations, take average:
            for i in range(evaluations):
                gal_imgs, query_imgs = get_rand_images(query, gallery)

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
                for fname in gal_imgs:
                    fpath = root + '/images/' + fname[0]

                    img = Image.open(fpath).convert('RGB')
                    img_t = test_transformer(img).cuda()
                    img_t = img_t.unsqueeze(0)
                    imgs_batch = torch.cat([imgs_batch, img_t], 0)

                vector_gal = self.model(imgs_batch)
                del img_t
                del imgs_batch

                # load query
                imgs_batch = Variable().cuda()
                for fname in query_imgs:
                    fpath = root + '/images/' + fname[0]

                    img = Image.open(fpath).convert('RGB')
                    img_t = test_transformer(img).cuda()
                    img_t = img_t.unsqueeze(0)

                    imgs_batch = torch.cat([imgs_batch, img_t], 0)

                vector_query = self.model(imgs_batch)
                del img_t
                del imgs_batch

                # calc distances
                m, n = vector_query.size(0), vector_gal.size(0)

                dist = torch.pow(vector_gal, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                       torch.pow(vector_query, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                dist.addmm_(1, -2, vector_gal, vector_query.t())

                query_ids = query_imgs[:,1]
                gallery_ids = gal_imgs[:,1]
                cmc, mAP = compute_accuracy(dist, query_ids, gallery_ids)
                # print("cmc: " + str(cmc) + "; mAP: " + str(mAP))

                # all = torch.empty(len(dist), 0)
                # all = [torch.sum(torch.stack([k < dist[i][i] for k in dist[i]])) for i in range(len(dist))]
                # all = torch.FloatTensor(all)

                top1_.append(cmc[0])
                top5_.append(cmc[4])
                top10_.append(cmc[9])
                mAP_.append(mAP)

            top1 = np.mean(top1_)
            top5 = np.mean(top5_)
            top10 = np.mean(top10_)
            mAP = np.mean(mAP_)
            print(name2save + "top1: " + str(top1))
            print(name2save + "top5: " + str(top5))
            print(name2save + "top10: " + str(top10))
            print(name2save + "mAP: " + str(mAP))



            if writer is not None:
                writer.add_scalar(name2save + 'Rank 1', top1, epoch)
                writer.add_scalar(name2save + 'Rank 5', top5, epoch)
                writer.add_scalar(name2save + 'Rank 10', top10, epoch)
                # writer.add_scalar(name2save + 'mAP', mAP, epoch)

            return top1

    # can be used to check if two vectors of the same person get mroe similar
    # function is not working for unsynchronized datasets
    def evaluate_one(self, query, gallery, print_freq, writer=None, epoch=None, root=None, height=None,
                             width=None, root2=None, name2save=""):



                # initialize loading and normalization
                normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

                test_transformer = T.Compose([
                    T.RectScale(height, width),
                    T.ToTensor(),
                    normalizer,
                ])

                fpath = root + '/images/' + query[0][0]

                img = Image.open(fpath).convert('RGB')
                img_t = test_transformer(img).cuda()
                img_t = img_t.unsqueeze(0)

                fpath = root2 + '/images/' + query[0][0]

                img = Image.open(fpath).convert('RGB')
                img_p = test_transformer(img).cuda()
                img_p = img_p.unsqueeze(0)

                print('retrain: ')
                print(self.model(img_t))
                print("orig: ")
                print(self.model_cm(img_p))


    def evaluate_single_shot_cm(self, query, gallery, print_freq, writer=None, epoch=None, root1=None, height=None,
                             width=None, root2=None, name2save=""):

        with torch.no_grad():
            torch.set_num_threads(1)

            top1_ = []
            top5_ = []
            top10_ = []
            mAP_ = []

            # 10 evaluations, take average:
            for i in range(30):
                query_imgs, gal_imgs = get_rand_images(query, gallery)#, imgs_from_q, imgs_from_g)

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
                for fname in query_imgs:
                    fpath = root1 + '/images/' + fname[0]

                    img = Image.open(fpath).convert('RGB')
                    img_t = test_transformer(img).cuda()
                    img_t = img_t.unsqueeze(0)
                    imgs_batch = torch.cat([imgs_batch, img_t], 0)

                self.model.eval()
                vector_gal = self.model(imgs_batch)

                del img_t
                del imgs_batch

                # load probe/query
                imgs_batch = Variable().cuda()
                for fname in gal_imgs:
                    fpath = root2 + '/images/' + fname[0]

                    img = Image.open(fpath).convert('RGB')
                    img_t = test_transformer(img).cuda()
                    img_t = img_t.unsqueeze(0)

                    imgs_batch = torch.cat([imgs_batch, img_t], 0)

                self.model_cm.eval()
                vector_query = self.model_cm(imgs_batch)

                del img_t
                del imgs_batch

                # calc distances
                m, n = vector_query.size(0), vector_gal.size(0)

                dist = torch.pow(vector_gal, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                       torch.pow(vector_query, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                dist.addmm_(1, -2, vector_gal, vector_query.t())

                query_ids = query_imgs[:,1]
                gallery_ids = gal_imgs[:,1]

                cmc, mAP = compute_accuracy(dist, query_ids, gallery_ids)
                # print("cmc: " + str(cmc) + "; mAP: " + str(mAP))

                # all = torch.empty(len(dist), 0)
                # all = [torch.sum(torch.stack([k < dist[i][i] for k in dist[i]])) for i in range(len(dist))]
                # all = torch.FloatTensor(all)

                top1_.append(cmc[0])
                top5_.append(cmc[4])
                top10_.append(cmc[9])
                mAP_.append(mAP)

            top1 = np.mean(top1_)
            top5 = np.mean(top5_)
            top10 = np.mean(top10_)
            mAP = np.mean(mAP_)
            print(name2save + "top1: " + str(top1))
            print(name2save + "top5: " + str(top5))
            print(name2save + "top10: " + str(top10))
            print(name2save + "mAP: " + str(mAP))


            if writer is not None:
                writer.add_scalar(name2save + 'Rank 1', top1, epoch)
                writer.add_scalar(name2save + 'Rank 5', top5, epoch)
                writer.add_scalar(name2save + 'Rank 10', top10, epoch)
                writer.add_scalar(name2save + 'mAP', mAP, epoch)

            return top1

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



# def cmc_partly(features,labels, writer, epoch):
#
#         x = torch.cat([features[f].unsqueeze(0) for f in features], 0)
#         y = x
#         m, n = x.size(0), y.size(0)
#         x = x.view(m, -1)
#         y = y.view(n, -1)
#
#         # this is a2-2ab+b2
#         dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#                torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#         dist.addmm_(1, -2, x, y.t())
#
#         labs = [labels[f] for f in labels]
#
#         matches = np.asarray([[(las==la).numpy() for la in labs] for las in labs])
#         sorted, indices = dist.sort()
#         cmc = matches[indices]
#
#         stands = [matches[i][indices[i]] for i in range (256)]
#         stands = np.array(stands)
#         cmc = [1, 2, 5]
#
#         acc1 = np.sum(stands[:,1])/len(stands)
#         acc5 = np.sum(np.any(stands[:, 1:5], 1)) / len(stands)
#         acc10 = np.sum(np.any(stands[:, 1:10], 1)) / len(stands)
#
#         if writer is not None:
#             writer.add_scalar('Rank 1', acc1, epoch)
#             writer.add_scalar('Rank 5', acc5, epoch)
#             writer.add_scalar('Rank 10', acc10, epoch)
#             #writer.add_scalar('mAP', mAP, epoch)
#
#         print("Accuracy 1: {}\n Accuracy 5: {}\n Accuracy 10: {}\n".format(acc1, acc5, acc10))
#
#         return acc1

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