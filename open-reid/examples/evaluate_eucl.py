import sys
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/')
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/utils')

import scipy.io as sio
import evaluators
import datasets
import os.path as osp
import h5py
import numpy as np
import glob

name1 = 'biwi'
name2 = 'biwi_depth'

data_dir = '/export/livia/data/FHafner/data'

root1 = osp.join(data_dir, name1)
root2 = osp.join(data_dir, name2)

dataset1 = datasets.create(name1, root1)
dataset2 = datasets.create(name2, root2)

list = '/export/livia/data/FHafner/data/' + name1 + '/images/'

list_all = glob.glob(list + '*')
list_all.sort()
list_all = [i.split('/')[-1] for i in list_all]
test_list = [i[0] for i in dataset1.query]


path = '/export/livia/home/vision/FHafner/MARS-evaluation/LOMO_XQDA/code/'
desc1 = path + 'descriptor_pku.mat'
desc2 = path + 'descriptor_pku_depth.mat'

# with h5py.File(desc2, 'r') as file:
#     a = list(file['descriptors'])
# pass
arrays = {}
f = h5py.File(desc1)
for k, v in f.items():
   arrays[k] = np.array(v)
pass
desc1 = arrays['descriptors']
del arrays
# print("desc1 loaded")
#
arrays = {}
f = h5py.File(desc2)
for k, v in f.items():
    arrays[k] = np.array(v)
desc2 = arrays['descriptors']
del arrays
# print("desc2 loaded")
# desc1 = sio.loadmat(desc1)
# desc2 = sio.loadmat(desc2)
# desc1 = desc1['desc1']
# desc2 = desc2['desc2']


te = 0
desc1_test = []
desc2_test = []
for ii in range(len(list_all)):
    if list_all[ii] in test_list:
        if int(sum(desc2[ii])) == 0:
            print('zero')
            pass

        desc1_test.append(desc1[ii])
        desc2_test.append(desc2[ii])
    if ii % 50 == 0:
        print(ii)
        print(sum(desc2[ii]))
del desc1
del desc2



#


dist = sio.loadmat('/export/livia/home/vision/FHafner/MARS-evaluation/LOMO_XQDA/code/dist.mat')
dist = dist['dist']

dataset1 = datasets.create(name1, root1)
dataset2 = datasets.create(name2, root2)

list = '/export/livia/data/FHafner/data/' + name1 + '/images/'

list_all = glob.glob(list + '*')

dist = dist[:len(dataset1.query),:len(dataset1.query)]

# query_ad, gallery_ad = evaluators.get_rand(dataset1.query, dataset2.gallery)

same = evaluators.getsame(dataset1.query, dataset2.gallery)
# query = [i[0] for i in dataset1.query]
#
# gallery = [i[0] for i in dataset2.query]

evaluators.evaluate_all(dist, query=dataset1.query, gallery=dataset2.gallery, same=same)
