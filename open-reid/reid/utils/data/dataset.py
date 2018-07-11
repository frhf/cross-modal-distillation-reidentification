from __future__ import print_function
import os.path as osp

import numpy as np

from ..serialization import read_json


def _pluck(identities, indices, cams, relabel=False):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                name = osp.splitext(fname)[0]
                x, y, _ = map(int, name.split('_'))

                assert pid == x and camid == y
                if cams == [] or np.any(cams == camid):
                    if relabel:
                        ret.append((fname, index, camid))
                    else:
                        ret.append((fname, pid, camid))

    return ret


class Dataset(object):
    def __init__(self, root, split_id=0):
        self.root = root
        self.split_id = split_id
        self.meta = None
        self.split = None
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0
        self.test_cam_gallery, self.test_cam_probe, self.train_cam = [], [], []

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')

    def load(self, num_val=0.3, load_val=False, cams=False, verbose=True):
        splits = read_json(osp.join(self.root, 'splits.json'))

        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))

        self.split = splits[self.split_id]

        query_pids = np.asarray(self.split['query'])
        gallery_pids = np.asarray(self.split['gallery'])

        # check if relevant cameras are given
        if cams:
            self.test_cam_gallery = np.asarray(self.split['test_cam_gallery'])
            self.test_cam_probe = np.asarray(self.split['test_cam_probe'])
            self.train_cam = np.asarray(self.split['train_cam'])
        else:
            self.test_cam_gallery = []
            self.test_cam_probe = []
            self.train_cam = []

        # implement load val
        if load_val:
            val_pids = np.asarray(self.split['val'])
            train_pids = np.asarray(self.split['train'])
            trainval_pids = np.asarray(self.split['trainval'])

            # exception for tum
            if len(trainval_pids)==149:
                val_pids -= 1
                train_pids -= 1
                trainval_pids -= 1
                gallery_pids -= 1
                query_pids -= 1

            np.random.shuffle(trainval_pids)
            num = len(trainval_pids)

            if isinstance(num_val, float):
                num_val = int(round(num * num_val))
            if num_val >= num or num_val < 0:
                raise ValueError("num_val exceeds total identities {}"
                                 .format(num))
        else:
            # Randomly split train / val
            trainval_pids = np.asarray(self.split['trainval'])
            np.random.shuffle(trainval_pids)
            num = len(trainval_pids)
            if isinstance(num_val, float):
                num_val = int(round(num * num_val))
            if num_val >= num or num_val < 0:
                raise ValueError("num_val exceeds total identities {}"
                                 .format(num))
            train_pids = sorted(trainval_pids[:-num_val])
            val_pids = sorted(trainval_pids[-num_val:])

        # cams are added here
        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']
        self.train = _pluck(identities, train_pids, self.train_cam, relabel=False)

        self.val_probe = _pluck(identities, val_pids, self.test_cam_probe, relabel=False)
        self.val_gallery = _pluck(identities, val_pids, self.test_cam_gallery, relabel=False)

        self.trainval = _pluck(identities, trainval_pids, self.train_cam, relabel=False)
        self.query = _pluck(identities, query_pids, self.test_cam_probe,)
        self.gallery = _pluck(identities, gallery_pids, self.test_cam_gallery,)
        self.num_train_ids = len(train_pids)
        self.num_val_ids = len(val_pids)
        self.num_trainval_ids = len(trainval_pids)

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_ids, len(self.train)))
            print("  val      | {:5d} | {:8d}"
                  .format(self.num_val_ids, len(self.val_probe)))
            print("  trainval | {:5d} | {:8d}"
                  .format(self.num_trainval_ids, len(self.trainval)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(self.split['query']), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(self.split['gallery']), len(self.gallery)))

    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json'))
