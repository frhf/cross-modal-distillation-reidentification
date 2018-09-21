import torch
import sys
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/')
sys.path.append('/export/livia/home/vision/FHafner/masterthesis/open-reid/reid/utils')

import datasets
import models
from dist_metric import DistanceMetric
from loss import TripletLoss
from trainers import Trainer
from evaluators import Evaluator
from utils.data import transforms as T
from utils.data.preprocessor import Preprocessor
from utils.data.sampler import RandomIdentitySampler
from utils.logging import Logger
from utils.serialization import load_checkpoint, save_checkpoint
from tensorboardX import SummaryWriter
import os
import cv2
import time
from PIL import Image
import torchvision

x1 = Image.open('/export/livia/home/vision/FHafner/images/onestream_sysu/comp_im1.png').convert('RGB')
x1 = x1.resize((128, 256))

# x1 = cv2.resize(x1, (256, 128))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
t = torchvision.transforms.ToTensor()
img = t(x1)

# img = torch.from_numpy(x1).float().to(device)
img = img.unsqueeze(0)
img = img.cuda()

models = models.create('resnet50', num_features=128, dropout=0, num_classes=50)
models = models.eval()
models = models.cuda()

timer = 0
for i in range(600):
    start = time.time()
    models(img)
    timer = timer + time.time()-start
print(timer/600)