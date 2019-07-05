import os
import sys
import glob
import cv2

from caffe_predict import *
from data_preprocession import AdaptImgForCaffeImgScheme

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

#constant
kPatchW = 225
kPatchH= 225

kNetDensityH = 28
kNetDensityW = 28

if __name__ == '__main__':
    InitCaffeEnv(os.path.expanduser('~/caffe/'), kHasGPU, kGPUId)
    net = LoadCaffeModel(kModelDef, kModelWeight)

    img = cv2.imread('1.jpg', 1)

    if kSliceSchemeOn:
        imgs = [img]
    else:
        imgs = AdaptImgForCaffeImgScheme([img])
    predict_densities = PredictImgsByCaffe(imgs, net, kOutLayer, kPatchW, kPatchH,
                                           kNetDensityW, kNetDensityH, kBatchSize,
                                           slice_scheme_on=kSliceSchemeOn)
    predict_density = predict_densities[0]
    print(predict_density)

# caffe_root = os.path.expanduser('~/caffe/') # change with your install location
# sys.path.insert(0, os.path.join(caffe_root, 'python'))
# sys.path.insert(0, os.path.join(caffe_root, 'python/caffe/proto'))
# import caffe
# import caffe_pb2
#
# model_name = 'dcc_crowdnet'
# model_path = os.path.expanduser(os.path.join('models', model_name))
# data_path = os.path.expanduser(os.path.join('data', model_name))
# weights_path = os.path.expanduser(os.path.join('weight', model_name))
#
# dataset_paths = ['dataset/UCF_CC_50']
#
# model_def = os.path.join(model_path, 'deploy_addCAFFE.prototxt')
#
# model_weights = glob.glob(os.path.join(weights_path, 'dcc_crowdnet_train_iter_144000.caffemodel'))[0]
#
# print("Get model_weight:")
# print(model_weights)
#
# print("Read in Net:")
# print("-----------------------------------------------")
# net = caffe.Net(model_def, model_weights, caffe.TEST)
