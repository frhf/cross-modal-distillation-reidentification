import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
from pylab import Rectangle

def calc_conf(indices, gallery_ids, query_ids, use_all, gallery_cams, query_cams, same):
    conf_mat = np.zeros((max(gallery_ids)+1, max(gallery_ids)+1))
    for i in range(len(indices)):
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                (gallery_cams[indices[i]] != query_cams[i]))# ATETNTNET

        if use_all:
            valid = same[i]

        valide_ind = indices[i, valid]
        conf_mat[query_ids[i], gallery_ids[valide_ind[0]]] += 1

    conf_mat = conf_mat[np.unique(gallery_ids)]
    conf_mat = conf_mat[:, np.unique(gallery_ids)]


    fig, ax = plt.subplots()
    ax.matshow(conf_mat, cmap=plt.cm.Blues)

    for i in range(len(conf_mat)):
        for j in range(len(conf_mat)):
            c = int(conf_mat[j, i])
            ax.text(i, j, str(c), va='center', ha='center')

    font = {'family': 'DejaVu Sans',
            'weight': 'normal',
            'size': 4}
    matplotlib.rc('font', **font)
    matplotlib.pyplot.xticks(np.arange(len(conf_mat)), np.unique(gallery_ids))
    matplotlib.pyplot.yticks(np.arange(len(conf_mat)), np.unique(gallery_ids))


    fig.savefig('/export/livia/home/vision/FHafner/images/conf.png', dpi=400)
    font = {'family': 'DejaVu Sans',
            'weight': 'normal',
            'size': 4}
    matplotlib.rc('font', **font)
    matplotlib.pyplot.xticks(np.arange(len(conf_mat)), np.unique(gallery_ids))
    matplotlib.pyplot.yticks(np.arange(len(conf_mat)), np.unique(gallery_ids))


    fig.savefig('/export/livia/home/vision/FHafner/images/conf.png', dpi=400)
    print("Confusion map calculated")



def make_comparison_img(root1, root2, distmat, query_ad, gallery_ad):
    for k in range(3):
        size_gallery = 6
        size_query = 4
        f, axarr = plt.subplots(size_query, size_gallery)
        for j in range(size_query):

            rnum = random.randint(0, len(distmat))
            # if query_ad[rnum][1] == gallery_ad[distmat[rnum].numpy().argsort()[:size_gallery][0]][1]:
            img_q = matplotlib.pyplot.imread(root1 + '/' + query_ad[rnum][0])
            axarr[j, 0].imshow(img_q)
            axarr[j, 0].set_title("Query", pad=2)
            axarr[j, 0].axis('off')
            curr = distmat[rnum]
            curr = curr.numpy()
            positions = curr.argsort()[:size_gallery]
            for i in range(size_gallery - 1):
                img_g = matplotlib.pyplot.imread(root2 + '/' + gallery_ad[positions[i]][0])
                axarr[j, i + 1].imshow(img_g)
                # set title
                axarr[j, i + 1].set_title("%.2f" % curr[positions[i]], pad=2)
                # set color
                if query_ad[rnum][1] == gallery_ad[positions[i]][1]:
                    color = 'green'
                else:
                    color = 'red'
                autoAxis = axarr[j, i + 1].axis()
                rec = Rectangle((autoAxis[0] - 0.7, autoAxis[2] - 0.2), (autoAxis[1] - autoAxis[0]) + 1,
                                (autoAxis[3] - autoAxis[2]) + 0.4, fill=False, lw=2, color=color)
                rec = axarr[j, i + 1].add_patch(rec)
                rec.set_clip_on(False)
                # turn off axis
                axarr[j, i + 1].axis('off')
                del rec
        # set font size
        font = {'family': 'DejaVu Sans',
                'weight': 'bold',
                'size': 5}

        matplotlib.rc('font', **font)
        f.savefig('/export/livia/home/vision/FHafner/images/comp_im' + str(k) + '.png', dpi=300)
    print("Comp image calculated")

def make_comparison_img_cm(root1, root2, distmat, query_ad, gallery_ad):
    for k in range(3):
        size_gallery = 6
        size_query = 4
        f, axarr = plt.subplots(size_query, size_gallery)
        for j in range(size_query):

            rnum = random.randint(0, len(distmat)-1)
            # if query_ad[rnum][1] == gallery_ad[distmat[rnum].numpy().argsort()[:size_gallery][0]][1]:
            img_q = matplotlib.pyplot.imread(root1 + '/' + query_ad[rnum][0])
            axarr[j, 0].imshow(img_q)
            axarr[j, 0].set_title("Query", pad=2)
            axarr[j, 0].axis('off')
            curr = distmat[rnum]
            curr = curr.numpy()
            positions = curr.argsort()[:size_gallery]
            for i in range(size_gallery - 1):
                img_g = matplotlib.pyplot.imread(root2 + '/' + gallery_ad[positions[i]][0])
                axarr[j, i + 1].imshow(img_g)
                # set title
                axarr[j, i + 1].set_title("%.2f" % curr[positions[i]], pad=2)
                # set color
                if query_ad[rnum][1] == gallery_ad[positions[i]][1]:
                    color = 'green'
                else:
                    color = 'red'
                autoAxis = axarr[j, i + 1].axis()
                rec = Rectangle((autoAxis[0] - 0.7, autoAxis[2] - 0.2), (autoAxis[1] - autoAxis[0]) + 1,
                                (autoAxis[3] - autoAxis[2]) + 0.4, fill=False, lw=2, color=color)
                rec = axarr[j, i + 1].add_patch(rec)
                rec.set_clip_on(False)
                # turn off axis
                axarr[j, i + 1].axis('off')
                del rec
        # set font size
        font = {'family': 'DejaVu Sans',
                'weight': 'bold',
                'size': 5}

        matplotlib.rc('font', **font)
        f.savefig('/export/livia/home/vision/FHafner/images/comp_im' + str(k) + '.png', dpi=300)
    print("Comp image calculated")

# indices = np.argsort(distmat, axis=1)
# gallery_ids = [x[1] for x in gallery_ad]
# gids = gallery_ids[indices[rnum][valid]]
# inds = np.where(valid)[0]
# ids_dict = defaultdict(list)
# for j, x in zip(inds, gids):
#     ids_dict[x].append(j)




    pass