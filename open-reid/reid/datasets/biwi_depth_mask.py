from __future__ import print_function, absolute_import
import os.path as osp

import numpy as np

import sys
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/utils/')

#from ..utils.data import Dataset
from utils.data import Dataset
from utils.osutils import mkdir_if_missing
from utils.serialization import write_json

class Biwi_depth_mask(Dataset):

    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(Biwi_depth_mask, self).__init__(root, split_id=split_id)

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(0.2)