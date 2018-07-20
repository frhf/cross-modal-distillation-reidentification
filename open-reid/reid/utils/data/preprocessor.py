from __future__ import absolute_import
import os.path as osp

from PIL import Image


class Preprocessor(object):
    def __init__(self, dataset1, dataset2=None, root=None, root2=None, transform=None):
        super(Preprocessor, self).__init__()
        if root2 is not None:
            self.dataset = dataset1 + dataset2

        elif root is not None:
            self.dataset = dataset1

        self.root = root
        self.root2 = root2

        self.dataset1_len = len(dataset1)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            if index < self.dataset1_len:
                fpath = osp.join(self.root, fname)
            else:
                fpath = osp.join(self.root2, fname)
                if self.root2 is None:
                    print("Error in root!")
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid


class PreprocessorRetrain(object):
    def __init__(self, dataset, root=None, transform=None):
        super(PreprocessorRetrain, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, enc = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, enc


# class PreprocessorOneStream(object):
#     def __init__(self, dataset1, dataset2, root1=None, root2=None, transform=None):
#         super(PreprocessorOneStream, self).__init__()
#         self.dataset = dataset1 + dataset2
#         self.root1 = root1
#         self.root2 = root2
#         self.dataset1_len = len(dataset1)
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, indices):
#         if isinstance(indices, (tuple, list)):
#             return [self._get_single_item(index) for index in indices]
#         return self._get_single_item(indices)
#
#     def _get_single_item(self, index):
#         fname, pid, camid = self.dataset[index]
#         fpath = fname
#         if self.root2 is not None:
#             if index < self.dataset1_len:
#                 fpath = osp.join(self.root1, fname)
#             else:
#                 fpath = osp.join(self.root2, fname)
#
#         img = Image.open(fpath).convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, fname, pid, camid