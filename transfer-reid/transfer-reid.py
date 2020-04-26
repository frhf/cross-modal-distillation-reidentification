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

sys.path.append('../open-reid/reid/')
sys.path.append('../open-reid/reid/utils')


def main(args):

    print("Start deep distillation")

    oldstr = 'f-' + args.from_ + '-t-' + args.to_ + '-' + args.name + '-split' + str(args.split_id) + '-val'
    name_val = oldstr.replace("/", "")
    name_val = osp.join(args.logdir,name_val)

    oldstr = 'f-' + args.from_ + '-t-' + args.to_ + '-' + args.name + '-split' + str(args.split_id) + '-test'
    name_test = oldstr.replace("/", "")
    name_test = osp.join(args.logdir,name_test)

    print("Groundtruth for distillation is saved in " + args.path_to_dist)

    path_to_origmodel = osp.join(args.path_to_orig, 'model_best.pth.tar')
    print("Model used as baseline for distillation is saved in " + path_to_origmodel)

    print("Distilled Model is saved to " + args.path_to_dist)

    if os.path.exists(args.path_to_dist) and not args.evaluate:
        raise Exception('There is already a trained model in the directory you are trying to save the retrained model to. Please delete it, if you want to save the new model to there.' )
        
    if not os.path.exists(args.path_to_dist):
        os.makedirs(args.path_to_dist)

    if args.extract:
        gtExtractor = GtExtractor(path_to_origmodel)
        if args.av:
            gtExtractor.extract_gt_av(args.from_, args.to_, path_to_save_gt=args.path_to_dist, extract_for='train')
            gtExtractor.extract_gt_av(args.from_, args.to_, path_to_save_gt=args.path_to_dist, extract_for='val')
            gtExtractor.extract_gt_av(args.from_, args.to_,  path_to_save_gt=args.path_to_dist, extract_for='query')

        else:
            gtExtractor.extract_gt(args.from_, data_dir=args.data_dir ,path_to_save_gt=args.path_to_dist, extract_for='train')
            gtExtractor.extract_gt(args.from_, data_dir=args.data_dir ,path_to_save_gt=args.path_to_dist, extract_for='val_gallery')
            gtExtractor.extract_gt(args.from_, data_dir=args.data_dir ,path_to_save_gt=args.path_to_dist, extract_for='val_probe')
            gtExtractor.extract_gt(args.from_, data_dir=args.data_dir ,path_to_save_gt=args.path_to_dist, extract_for='gallery')
            gtExtractor.extract_gt(args.from_, data_dir=args.data_dir ,path_to_save_gt=args.path_to_dist, extract_for='query')

    retrainer = Retrainer(path_to_origmodel, dropout=args.dropout, freeze_model=args.freeze_model, load_weights=args.load_weights)

    retrainer.retrain(args.to_, args.from_, args.path_to_dist, root=args.data_dir, batch_size=args.batch_size, epochs=args.epochs, combine_trainval=False, workers=args.workers,
                      path_to_retmodel=args.path_to_dist, evaluate=args.evaluate, split_id=args.split_id, name_val=name_val,
                      name_test=name_test)

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transfer reid")
    parser.add_argument('-f', '--from_', type=str, default='biwi_depth')
    parser.add_argument('-t', '--to_', type=str, default='biwi')
    parser.add_argument('--extract', type=bool, default=False)
    parser.add_argument('--av', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--logdir', type=str, default='../../logdir')
    parser.add_argument('--path-to-orig', type=str)
    parser.add_argument('--path-to-dist', type=str)    
    parser.add_argument('--name', type=str, default='dummy_name')
    parser.add_argument('--split-id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)    
    parser.add_argument('--dropout', type=int, default=0.5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--data-dir', type=str, default='../../data')


    # feature_parser = parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('--dont-freeze-model', dest='freeze_model', action='store_false')
    parser.add_argument('--dont-load-weights', dest='load_weights', action='store_false')
    parser.set_defaults(freeze_model=True)
    parser.set_defaults(load_weights=True)


    main(parser.parse_args())
