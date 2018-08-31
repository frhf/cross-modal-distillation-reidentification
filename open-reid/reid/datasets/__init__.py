from __future__ import absolute_import
import warnings

from .cuhk01 import CUHK01
from .cuhk03 import CUHK03
from .dukemtmc import DukeMTMC
from .market1501 import Market1501
from .viper import VIPeR
from .biwi import Biwi
from .biwi_depth import Biwi_depth
from .biwi_depth_mask import Biwi_depth_mask
from .somaset import Somaset
from .tum import Tum
from .tum_depth import Tum_depth
from .tum_comb import Tum_comb
from .sysu import Sysu
from .sysu_ir import Sysu_ir
from .iit import Iit
from .iit_depth import Iit_depth
from .iit_depth_pc import Iit_depth_pc
from .pku import Pku
from .pku_depth import Pku_depth
from .synthia import Synthia
from .synthia_depth import Synthia_depth


__factory = {
    'viper': VIPeR,
    'cuhk01': CUHK01,
    'cuhk03': CUHK03,
    'market1501': Market1501,
    'dukemtmc': DukeMTMC,
    'biwi': Biwi,
    'biwi_depth': Biwi_depth,
    'biwi_depth_mask': Biwi_depth_mask,
    'somaset': Somaset,
    'tum': Tum,
    'tum_depth': Tum_depth,
    'tum_comb': Tum_comb,
    'sysu': Sysu,
    'sysu_ir': Sysu_ir,
    'iit': Iit,
    'iit_depth': Iit_depth,
    'iit_depth_pc': Iit_depth_pc,
    'pku': Pku,
    'pku_depth': Pku_depth,
    'synthia': Synthia,
    'synthia_depth': Synthia_depth
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
