import os
import sys
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import copy
from skimage import measure
from skimage.transform import resize

import importlib

from unet import UNet
from resnet import ResNet

from scripts.functions import box_locations, train, iou_box
from scripts.imgProcessor import imgProcessor

sys.path.append(os.path.join('../model'))

locations = box_locations()

train_dire = "../data/stage_2_train_images"
files = os.listdir(train_dire)
# random.shuffle(files)
num_validation = 2560
train_d = files[num_validation:]
validation_d = files[:num_validation]

BATCH_SIZE = 16
IMAGE_SIZE = 320
EPOCH_NUM = 5

model = UNet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
# model.compile(optimizer='adam', loss=nn.functional.)

gpu_dtype = torch.cuda.FloatTensor
fixed_model_gpu = copy.deepcopy(model).type(gpu_dtype)
loss_fn = nn.BCELoss().type(gpu_dtype)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_dire = os.path.join(train_dire)
train_ip = imgProcessor(train_dire, train_d, locations, shuffle=True, aug=True, predict=False)
valid_ip = imgProcessor(train_dire, validation_d, locations, shuffle=False, aug=False, predict=False)
test_ip = imgProcessor(train_dire, validation_d, None, shuffle=False, predict=True)

train(train_ip, gpu_dtype, model, loss_fn, optimizer, EPOCH_NUM)

prob_thresholds = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
nthresh = len(prob_thresholds)

count = 0
ns = nthresh * [0]
nfps = nthresh * [0]
ntps = nthresh * [0]
overall_maps = nthresh * [0.]
for imgs, filenames in test_ip:
    # predict batch of images
    preds = []
    for x, y in imgs:
        with torch.no_grad():
            x_var = Variable(x.type(gpu_dtype))
        scores = model(x_var)
        _, pred = scores.data.cpu().max(1)
        preds.append(pred)
    # preds = model.predict(imgs)
    # loop through batch
    for pred, filename in zip(preds, filenames):
        count = count + 1
        maxpred = np.max(pred)
        # resize predicted mask
        pred = resize(pred, (1024, 1024), mode='reflect')
        # threshold predicted mask
        boxes_preds = []
        scoress = []
        for thresh in prob_thresholds:
            comp = pred[:, :, 0] > thresh
            # apply connected components
            comp = measure.label(comp)
            # apply bounding boxes
            boxes_pred = np.empty((0, 4), int)
            scores = np.empty((0))
            for region in measure.regionprops(comp):
                y, x, y2, x2 = region.bbox
                boxes_pred = np.append(boxes_pred, [[x, y, x2 - x, y2 - y]], axis=0)
                conf = np.mean(pred[y:y2, x:x2])
                scores = np.append(scores, conf)
            boxes_preds = boxes_preds + [boxes_pred]
            scoress = scoress + [scores]
        boxes_true = np.empty((0, 4), int)
        fn = filename.split('.')[0]

        # if image contains pneumonia
        if fn in box_locations:
            # loop through pneumonia
            for location in box_locations[fn]:
                x, y, w, h = location
                boxes_true = np.append(boxes_true, [[x, y, w, h]], axis=0)
        for i in range(nthresh):
            if boxes_true.shape[0] == 0 and boxes_preds[i].shape[0] > 0:
                ns[i] = ns[i] + 1
                nfps[i] = nfps[i] + 1
            elif boxes_true.shape[0] > 0:
                ns[i] = ns[i] + 1
                contrib = iou_box(boxes_true, boxes_preds[i], scoress[i])
                overall_maps[i] = overall_maps[i] + contrib
                if boxes_preds[i].shape[0] > 0:
                    ntps[i] = ntps[i] + 1

    if count >= len(validation_d):
        break

for i, thresh in enumerate(prob_thresholds):
    print("\nProbability threshold=", thresh)
    overall_maps[i] = overall_maps[i] / (ns[i] + 1e-7)
    print("False positive cases=", nfps[i])
    print("True positive cases=", ntps[i])
    print("Overall evaluation score=", overall_maps[i])

# save model
model_path = 'model' + datetime.now().strftime("%Y%m%d_%H:%M:%S") + '.hdf5'
model.save(os.path.join(model_path))