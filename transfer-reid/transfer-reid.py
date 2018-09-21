from __future__ import print_function, absolute_import
import argparse
import os.path as osp


import sys
from gt_extractor import GtExtractor
from retrainer import Retrainer
import datasets
import os
import os.path as osp
import numpy as np

sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/')
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/utils')


def main(args):
    # THINGS TO CHANGE:
    # from_, to_, path to save, if new net: path to net


    # path_to_basemodel = logdir + 'sysu_ir/train/triplet_resnet18/model_best.pth.tar'
    # path_to_save_gt = logdir + 'sysu_ir/train/triplet_resnet18/'
    # path_to_basemodel = logdir + 'sysu/triplet_resnet18/model_best.pth.tar'
    # path_to_save_gt = logdir + 'sysu/triplet_resnet18/'

    # TUM
    # path_to_basemodel = '//export/livia/data/FHafner/data/logdir/tum_depth/retrain/triplet_resnet18_control/' \
    #                     'model_best.pth.tar'
    # path_to_save_gt = '/export/livia/data/FHafner/data/logdir/tum_depth/retrain/triplet_resnet18_control/'

    # SYSU
    # path_to_basemodel = '/export/livia/data/FHafner/data/logdir/sysu/triplet_resnet18/' \
    #                  'model_best.pth.tar'
    # path_to_save_gt = '/export/livia/data/FHafner/data/logdir/sysu/triplet_resnet18/'
    #
    # SYSU_IR
    # path_to_basemodel = '/export/livia/data/FHafner/data/logdir/sysu_ir/triplet_resnet18/' \
    #                  'model_best.pth.tar'
    # path_to_save_gt = '/export/livia/data/FHafner/data/logdir/sysu_ir/triplet_resnet18/'
    print("start")
    # from_ = 'tum_depth'
    # to_ = 'tum'

    # from_ = 'tum'
    # to_ = 'tum_depth'
    oldstr = 'f-' + args.from_ + '-t-' + args.to_ + '-' + args.name + '-split' + str(args.split_id) + '-val'
    name_val = oldstr.replace("/", "")

    oldstr = 'f-' + args.from_ + '-t-' + args.to_ + '-' + args.name + '-split' + str(args.split_id) + '-test'
    name_test = oldstr.replace("/", "")
    print(name_test)

    path_to_save_gt = osp.join(args.logdir, args.from_, args.path_to_orig)
    path_to_origmodel = osp.join(path_to_save_gt, 'model_best.pth.tar')


    path_to_retsavemodel = osp.join(args.logdir, args.to_, 'retraining/', args.name)
    # path_to_retstartmodel = osp.join(logdir, to_, 'retraining/softmax/same_unfrozen_av_ep50/model_best.pth.tar')



    if os.path.exists(path_to_retsavemodel) and not args.evaluate:
        raise Exception('There is already a trained model in the directory:' + path_to_retsavemodel)

    if args.extract:
        gtExtractor = GtExtractor(path_to_origmodel)
        if args.av:
            gtExtractor.extract_gt_av(args.from_, args.to_, path_to_save_gt=path_to_save_gt, extract_for='train')
            gtExtractor.extract_gt_av(args.from_, args.to_, path_to_save_gt=path_to_save_gt, extract_for='val')
            gtExtractor.extract_gt_av(args.from_, args.to_,  path_to_save_gt=path_to_save_gt, extract_for='query')

        else:
            gtExtractor.extract_gt(args.from_, path_to_save_gt=path_to_save_gt, extract_for='train')
            gtExtractor.extract_gt(args.from_, path_to_save_gt=path_to_save_gt, extract_for='val_gallery')
            gtExtractor.extract_gt(args.from_, path_to_save_gt=path_to_save_gt, extract_for='val_probe')
            gtExtractor.extract_gt(args.from_, path_to_save_gt=path_to_save_gt, extract_for='gallery')
            gtExtractor.extract_gt(args.from_, path_to_save_gt=path_to_save_gt, extract_for='query')

    if not os.path.exists(path_to_retsavemodel):
        os.makedirs(path_to_retsavemodel)

    retrainer = Retrainer(path_to_origmodel, dropout=0.5, freeze_model=args.freeze_model, load_weights=args.load_weights)
    # retrainer = Retrainer(path_to_retstartmodel, path_to_origmodel, dropout=0.3, freeze_model=True)

    retrainer.retrain(args.to_, args.from_, path_to_save_gt, batch_size=64, epochs=30, combine_trainval=False, workers=3,
                      path_to_retmodel=path_to_retsavemodel, evaluate=args.evaluate, split_id=args.split_id, name_val=name_val,
                      name_test=name_test)
    # retrainer.re_evaluate_retrain(to_, from_, path_to_save_gt, batch_size=64, combine_trainval=False,
    #                            workers=3, path_to_retsavemodel=path_to_retsavemodel)

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transfer reid")
    parser.add_argument('-f', '--from_', type=str, default='biwi_depth')
    parser.add_argument('-t', '--to_', type=str, default='biwi')
    parser.add_argument('--extract', type=bool, default=False)
    parser.add_argument('--av', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--logdir', type=str, default='/export/livia/data/FHafner/data/logdir/')
    parser.add_argument('--path-to-orig', type=str) # 'train/triplet-resnet18/'
    parser.add_argument('--name', type=str) # 'triplet/same_ep_av/'
    parser.add_argument('--split-id', type=int, default=0)

    # feature_parser = parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('--dont-freeze-model', dest='freeze_model', action='store_false')
    parser.add_argument('--dont-load-weights', dest='load_weights', action='store_false')
    parser.set_defaults(freeze_model=True)
    parser.set_defaults(load_weights=True)


    # parser.add_argument('--freeze-model', type=bool, action='store_true')
    # parser.add_argument('--load-weights', type=bool, action='store_false')


    main(parser.parse_args())
