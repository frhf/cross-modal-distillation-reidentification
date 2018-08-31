from __future__ import print_function, absolute_import
import os.path as osp

import numpy as np

import sys
sys.path.append('..')


#from ..utils.data import Dataset
from utils.data import Dataset
from utils.osutils import mkdir_if_missing
from utils.serialization import write_json

class Iit_depth(Dataset):

    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(Iit_depth, self).__init__(root, split_id=split_id)

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(load_val=True, cams=False)