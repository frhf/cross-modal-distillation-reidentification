from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable



import sys
sys.path.append('../reid/')

from evaluation_metrics import accuracy
from loss import TripletLoss
from utils.meters import AverageMeter
import numpy as np


class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        model.cuda()
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=1, writer=None):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        # solution to problem that data in retraining is different
        if not isinstance(data_loader.dataset.dataset[0][1], np.ndarray):
            x = list(set([data[1] for data in data_loader.dataset.dataset]))
            y = [n for n in range(len(x))]
            num_dict = dict(zip(x, y))
        else:
            num_dict = None

        end = time.time()
        for i, inputs in enumerate(data_loader):

            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs, num_dict)
            loss, prec1 = self._forward(inputs, targets)

            if isinstance(self.criterion, torch.nn.MSELoss):
                loss_show = torch.sqrt(loss).sum() / len(inputs[0])
            else:
                loss_show = loss

            losses.update(loss_show.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if writer is not None:
                writer.add_scalar('loss', loss_show, i+epoch*len(data_loader))

            if prec1 != 0:
                if (i + 1) % print_freq == 0:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Data {:.3f} ({:.3f})\t'
                          'Loss {:.3f} ({:.3f})\t'
                          'Prec {:.2%} ({:.2%})\t'
                          .format(epoch, i + 1, len(data_loader),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses.val, losses.avg,
                                  precisions.val, precisions.avg))

            else:
                if (i + 1) % print_freq == 0:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Data {:.3f} ({:.3f})\t'
                          'Loss {:.3f} ({:.3f})\t'

                          .format(epoch, i + 1, len(data_loader),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses.val, losses.avg))


    def train_twostream(self, epoch, data_loader1, data_loader2, optimizer, print_freq=1, writer=None):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        # dirty solution to problem that data in retraining is different
        if not isinstance(data_loader1.dataset.dataset[0][1], np.ndarray):
            x = list(set([data[1] for data in data_loader1.dataset.dataset]))
            y = [n for n in range(len(x))]
            num_dict = dict(zip(x, y))
        else:
            num_dict = None

        # dirty solution to problem that data in retraining is different
        if not isinstance(data_loader2.dataset.dataset[0][1], np.ndarray):
            x = list(set([data[1] for data in data_loader2.dataset.dataset]))
            y = [n for n in range(len(x))]
            num_dict = dict(zip(x, y))
        else:
            num_dict = None

        end = time.time()
        for i, (inputs1, inputs2) in enumerate(zip(data_loader1, data_loader2)):

            data_time.update(time.time() - end)

            #print("RGB here")
            inputs1, targets1 = self._parse_data(inputs1, num_dict)
            loss1, prec1 = self._forward(inputs1, targets1)

            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()

            #print("Depth here")
            inputs2, targets2 = self._parse_data(inputs2, num_dict)
            loss2, prec2 = self._forward(inputs2, targets2)



            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()

            loss = loss1 + loss2            

            if isinstance(self.criterion, torch.nn.MSELoss):
                loss_show = torch.sqrt(loss).sum() / (len(inputs1[0])+len(inputs2[0]))
            else:
                loss_show = loss

            losses.update(loss_show.data[0], targets1.size(0))
            precisions.update((prec1 + prec2) / 2, targets1.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if writer is not None:
                writer.add_scalar('loss', loss_show, i+epoch*len(data_loader1))

            if prec1 != 0:
                if (i + 1) % print_freq == 0:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Data {:.3f} ({:.3f})\t'
                          'Loss {:.3f} ({:.3f})\t'
                          'Prec {:.2%} ({:.2%})\t'
                          .format(epoch, i + 1, len(data_loader1),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses.val, losses.avg,
                                  precisions.val, precisions.avg))

            else:
                if (i + 1) % print_freq == 0:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Data {:.3f} ({:.3f})\t'
                          'Loss {:.3f} ({:.3f})\t'

                          .format(epoch, i + 1, len(data_loader2),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses.val, losses.avg))



    def _parse_data(self, inputs, num_dict=None):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError

class Trainer(BaseTrainer):
    def _parse_data(self, inputs, num_dict=None):
        imgs, _, pids, _ = inputs

        pids_np = np.array(pids)
        pids_dict = torch.tensor([num_dict[pi] for pi in pids_np], dtype=torch.int64)
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids_dict.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
	
        outputs = self.model(*inputs)#, origin)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec


class TrainerRetrainer(BaseTrainer):
    def _parse_data(self, inputs, num_dict=None):
        imgs, name, enc = inputs
        inputs = [Variable(imgs)]
        targets = Variable(enc.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        self.model.train()
        inputs = inputs[0].cuda()
        outputs = self.model(inputs)

        loss = self.criterion(outputs, targets)

        return loss, 0
