import sys
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/')
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/utils')

import scipy.io as sio
import evaluators
import datasets
import os.path as osp


name1 = 'biwi'
name2 = 'biwi_depth'

data_dir = '/export/livia/data/FHafner/data'

root1 = osp.join(data_dir, name1)
root2 = osp.join(data_dir, name2)

dist = sio.loadmat('/export/livia/data/FHafner/data/pku/dist_biwi_depth_PAMIeucl.mat')
dist = dist['dist']

dataset1 = datasets.create(name1, root1)
dataset2 = datasets.create(name2, root2)

dist = dist[:len(dataset1.query),:len(dataset1.query)]

# query_ad, gallery_ad = evaluators.get_rand(dataset1.query, dataset2.gallery)

same = evaluators.getsame(dataset1.query, dataset2.gallery)
# query = [i[0] for i in dataset1.query]
#
# gallery = [i[0] for i in dataset2.query]

evaluators.evaluate_all(dist, query=dataset1.query, gallery=dataset2.gallery, same=same)
