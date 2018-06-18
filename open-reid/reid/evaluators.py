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



# evaluates NN and saves time for doing so
def extract_features(model, data_loader, print_freq=1, metric=None, n_batches=None):
    # evaluation mode for model
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

        if (i + 1) % print_freq == 0:
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
                 cmc_topk=(1, 2, 5), writer=None, epoch=None):
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
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))
    print("mAP: " + str(time.time() - beg))

    # Compute all kinds of CMC scores
    cmc_configs = {
        # 'allshots': dict(separate_camera_set=False,
        #                  single_gallery_shot=False,
        #                  first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
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

    if writer is not None:
        writer.add_scalar('Rank 1', cmc_scores['cuhk03'][0], epoch)
        writer.add_scalar('Rank 2', cmc_scores['cuhk03'][1], epoch)
        writer.add_scalar('Rank 5', cmc_scores['cuhk03'][4], epoch)
        writer.add_scalar('mAP', mAP, epoch)

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores['cuhk03'][0]


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, print_freq, writer=None, epoch=None, metric=None):
        features, _ = extract_features(self.model, data_loader, print_freq)
        distmat = pairwise_distance(features, query, gallery, metric=metric)
        return evaluate_all(distmat, query=query, gallery=gallery, writer=writer, epoch=epoch)

    def evaluate_retrain(self, val_loader_ret, val_loader_int, criterion, epoch, query, gallery, writer=None):

        self.model.eval()
        overall_loss = 0
        for batch in val_loader_ret:
            imgs = Variable(batch[0].cuda(), requires_grad=False)
            out = self.model(imgs)
            target = Variable(batch[2].cuda(), requires_grad=False)
            loss = Variable(criterion(out, target), requires_grad=False)
            overall_loss += loss

            # free memory
            imgs = None
            out = None
            target = None

        overall_loss_n = overall_loss.cpu().numpy()
        print('Validation Loss: ' + str(overall_loss_n))

        if writer is not None:
            writer.add_scalar('ValidationLoss', overall_loss_n, epoch)


        self.evaluate(val_loader_int, query, gallery, 1, dataset, writer, epoch)


    def evaluate_partly(self, data_loader, query, gallery, print_freq, writer=None, epoch=None, n_batches=None, metric=None):
        features, labels = extract_features(self.model, data_loader, print_freq, n_batches=n_batches)
        acc1 = cmc_partly(features, labels, writer, epoch)
        return acc1


    def evaluate_single_shot(self, data_loader, query, gallery, print_freq, writer=None, epoch=None, root=None, height=None, width=None):

        with torch.no_grad():
            torch.set_num_threads(1)

            self.model.eval()

            top1_ = []
            top5_ = []
            top10_ = []

            # 10 evaluations, take average:
            for i in range(5):
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

                all = torch.empty(len(dist), 0)
                all = [torch.sum(torch.stack([k < dist[i][i] for k in dist[i]])) for i in range(len(dist))]
                all = torch.FloatTensor(all)

                top1_.append(len(all[all < 1])/len(dist))
                top5_.append(len(all[all < 6])/len(dist))
                top10_.append(len(all[all < 11])/len(dist))

            top1 = np.mean(top1_)
            top5 = np.mean(top5_)
            top10 = np.mean(top10_)
            print("top1: " + str(top1))
            print("top5: " + str(top5))
            print("top10: " + str(top10))


            if writer is not None:
                writer.add_scalar('Rank 1', top1, epoch)
                writer.add_scalar('Rank 2', top5, epoch)
                writer.add_scalar('Rank 5', top10, epoch)

            return top1



def cmc_partly(features,labels, writer, epoch):

        x = torch.cat([features[f].unsqueeze(0) for f in features], 0)
        y = x
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)

        # this is a2-2ab+b2
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, x, y.t())

        labs = [labels[f] for f in labels]

        matches = np.asarray([[(las==la).numpy() for la in labs] for las in labs])
        sorted, indices = dist.sort()
        cmc = matches[indices]

        stands = [matches[i][indices[i]] for i in range (256)]
        stands = np.array(stands)
        cmc = [1, 2, 5]

        acc1 = np.sum(stands[:,1])/len(stands)
        acc5 = np.sum(np.any(stands[:, 1:5], 1)) / len(stands)
        acc10 = np.sum(np.any(stands[:, 1:10], 1)) / len(stands)

        if writer is not None:
            writer.add_scalar('Rank 1', acc1, epoch)
            writer.add_scalar('Rank 5', acc5, epoch)
            writer.add_scalar('Rank 10', acc10, epoch)
            #writer.add_scalar('mAP', mAP, epoch)

        print("Accuracy 1: {}\n Accuracy 5: {}\n Accuracy 10: {}\n".format(acc1, acc5, acc10))

        return acc1

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

    return gal_imgs, query_imgs

