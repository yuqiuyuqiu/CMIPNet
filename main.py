import torch
import numpy as np

import time
import torch.nn as nn
from torch.nn import init
import os
import sys
import cv2
from utils.model_init import model_init
from framework import Framework
from utils.datasets import prepare_Beijing_dataset, prepare_TLCGIS_dataset

from networks.CMIP_dlink34net import DinkNet34_CMIPNet
from networks.hrnet18 import CMIP_hrnet18

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def get_model(model_name):
    if model_name == 'CMIPNet_Dlinknet34':
       model =DinkNet34_CMIPNet()
       model_init(model, 'resnet34', 2, imagenet=True)
    elif model_name == 'CMIPNet_HRNet':
       model =CMIP_hrnet18(pretrained=False)
    else:
        print("[ERROR] can not find model ", model_name)
        assert(False)
    return model

def get_dataloader(args):
    if args.dataset =='BJRoad':
        train_ds, val_ds, test_ds = prepare_Beijing_dataset(args) 
    elif args.dataset == 'TLCGIS' or args.dataset.find('Porto') >= 0:
        train_ds, val_ds, test_ds = prepare_TLCGIS_dataset(args) 
    else:
        print("[ERROR] can not find dataset ", args.dataset)
        assert(False)  

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=True,  drop_last=False)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False, drop_last=False)
    test_dl  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False, drop_last=False)
    return train_dl, val_dl, test_dl


def train_val_test(args):
    net = get_model(args.model)
    #print(net)

    print('lr:',args.lr)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)

    framework = Framework(net, optimizer, dataset=args.dataset)
    
    train_dl, val_dl, test_dl = get_dataloader(args)
    framework.set_train_dl(train_dl)
    framework.set_validation_dl(val_dl)
    framework.set_test_dl(test_dl)
    framework.set_save_path(WEIGHT_SAVE_DIR)

    framework.fit(cos_lr=args.cos_lr,lam=args.lam,t=args.t,epochs=args.epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CMIPNet_Dlinknet34')
    parser.add_argument('--lr',    type=float, default=2e-4)
    parser.add_argument('--name',  type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sat_dir',  type=str, default=r'E:\ML_data\remote_data\BJRoad\train_val\image')
    parser.add_argument('--mask_dir', type=str, default=r'E:\ML_data\remote_data\BJRoad\train_val\mask')
    parser.add_argument('--gps_dir',  type=str, default=r'E:\ML_data\remote_data\BJRoad\train_val\gps')
    parser.add_argument('--test_sat_dir',  type=str, default=r'E:\ML_data\remote_data\BJRoad\test\image')
    parser.add_argument('--test_mask_dir', type=str, default=r'E:\ML_data\remote_data\BJRoad\test\mask')
    parser.add_argument('--test_gps_dir',  type=str, default=r'E:\ML_data\remote_data\BJRoad\test\gps')
    # parser.add_argument('--sat_dir', type=str, default='/home/nku/hdd/data/BJRoad/BJRoad/train_val/image')
    # parser.add_argument('--mask_dir', type=str, default='/home/nku/hdd/data/BJRoad/BJRoad/train_val/mask')
    # parser.add_argument('--gps_dir', type=str, default='/home/nku/hdd/data/BJRoad/BJRoad/train_val/gps')
    # parser.add_argument('--test_sat_dir', type=str, default='/home/nku/hdd/data/BJRoad/BJRoad/test/image')
    # parser.add_argument('--test_mask_dir', type=str, default='/home/nku/hdd/data/BJRoad/BJRoad/test/mask')
    # parser.add_argument('--test_gps_dir', type=str, default='/home/nku/hdd/data/BJRoad/BJRoad/test/gps')

    parser.add_argument('--lidar_dir',  type=str, default='')
    parser.add_argument('--split_train_val_test', type=str, default='')
    parser.add_argument('--weight_save_dir', type=str, default='./save_model')
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--use_gpu',  type=bool, default=True)
    parser.add_argument('--gpu_ids',  type=str, default='0')
    parser.add_argument('--workers',  type=int, default=0)
    parser.add_argument('--epochs',  type=int, default=30)
    parser.add_argument('--lam', type=float, default=3e-4)
    parser.add_argument('--t', type=float, default=0.6)
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--dataset', type=str, default='BJRoad')
    parser.add_argument('--down_scale', type=bool, default=True)
    parser.add_argument('--cos_lr', type=bool, default=True)
    args = parser.parse_args()

    if args.use_gpu:
        try:
            gpu_list = [int(s) for s in args.gpu_ids.split(',')]
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
        BATCH_SIZE = args.batch_size * len(gpu_list)
    else:
        BATCH_SIZE = args.batch_size
        
    WEIGHT_SAVE_DIR = os.path.join(args.weight_save_dir, f"{args.model}_{args.dataset}_"+time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())+"/")
    if not os.path.exists(WEIGHT_SAVE_DIR):
        os.makedirs(WEIGHT_SAVE_DIR)
    print("Log dir: ", WEIGHT_SAVE_DIR)
    
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(WEIGHT_SAVE_DIR+'train.log')

    train_val_test(args)
    print("[DONE] finished")

