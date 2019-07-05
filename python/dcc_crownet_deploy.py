'''
@Date: 2019-05-15 05:10:14
@Author: Xiaobo Yang
@Email: hal_42@zju.edu.cn
@Company: Zhejiang University
@LastEditors: Xiaobo Yang
@LastEditTime: 2019-05-16 10:48:55
@Description: 
'''
#!/usr/bin/env python
# coding: utf-8

from data_preprocession import *
from caffe_predict import *

import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import time
from multiprocessing import pool

#caffe
kCaffeRoot = os.path.expanduser('~/caffe/') # change with your install location
kModelName = 'dcc_crowdnet'
kModelPath = os.path.expanduser(os.path.join('..','models', kModelName))
kDataPath = os.path.expanduser(os.path.join('..','data', kModelName))
kWeightsPath = os.path.expanduser(os.path.join('..','weight', kModelName))
kModelWeight = os.path.join(kWeightsPath, 'dcc_crowdnet_train_iter_139000.caffemodel')
kModelDef = os.path.join(kModelPath, 'deploy_addCAFFE.prototxt')

kOutLayer = 'conv6'
kBatchSize = 40
kSliceSchemeOn = True

kHasGPU = True
kGPUId = 0

#dataset
kDatasetPaths = [os.path.join('..', 'dataset', 'UCF_CC_50')]

#constant
kPatchW = 225
kPatchH= 225

kNetDensityH = 28
kNetDensityW = 28

#debug
kIsOnline = True
kSaveTest = True

def _MultiProcessPredict(img):
    InitCaffeEnv(os.path.expanduser('~/caffe/'), kHasGPU, kGPUId)
    net = LoadCaffeModel(kModelDef, kModelWeight)
    predict_densities = PredictImgsByCaffe([img], net, kOutLayer, kPatchW, kPatchH,
                                           kNetDensityW, kNetDensityH, kBatchSize,
                                           slice_scheme_on=kSliceSchemeOn)
    return predict_densities[0]

def CaffeTestDataset(path: str) -> int:
    raw_imgs, _, gt_densities, img_paths = load_images_and_gts(path)
    if kIsOnline:
        if kSliceSchemeOn:
            InitCaffeEnv(os.path.expanduser('~/caffe/'), kHasGPU, kGPUId)
            net = LoadCaffeModel(kModelDef, kModelWeight)
            predict_densities = PredictImgsByCaffe(raw_imgs, net, kOutLayer, kPatchW, kPatchH,
                                                   kNetDensityW, kNetDensityH, kBatchSize,
                                                   slice_scheme_on=kSliceSchemeOn)
        else:
            print("Not an Good Idea to Use this Funciton!")
            imgs = AdaptImgForCaffeImgScheme(raw_imgs)
            predict_densities = []
            for img in imgs:
                pool = Pool(processes=1)
                res = pool.apply_async(_MultiProcessPredict, (img,))
                pool.close()
                pool.join()
                predict_densities.append(res.get())
    else:
        predict_densities = DummyPredictImgsByCaffe(raw_imgs)
    ShowTestResult(predict_densities, raw_imgs, gt_densities, img_paths,
                    is_save=kSaveTest, path=path)
    return 0


if __name__ == '__main__':
    for i, dataset_path in enumerate(kDatasetPaths):
        print("Testing " + str(i) +"th dataset at" + dataset_path)
        CaffeTestDataset(dataset_path)