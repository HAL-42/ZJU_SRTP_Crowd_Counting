#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: server.py.py
@time: 2019/5/18 10:10
@desc:
"""


import socketserver
import cv2
import numpy as np
import os
from caffe_predict import *
from socket_send_recv import *
import time
import logging
from multiprocessing import Pool

from typing import Optional

encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
kHeadLen = 16

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

kLoop = 10

#debug
kIsOnline = True


def _MultiProcessPredict(img):
    InitCaffeEnv(os.path.expanduser('~/caffe/'), kHasGPU, kGPUId)
    net = LoadCaffeModel(kModelDef, kModelWeight)
    if kSliceSchemeOn:
        imgs = [img]
    else:
        print("Not an Good Idea to Use this Function")
        imgs = AdaptImgForCaffeImgScheme([img])
    predict_densities = PredictImgsByCaffe(imgs, net, kOutLayer, kPatchW, kPatchH,
                                           kNetDensityW, kNetDensityH, kBatchSize,
                                           slice_scheme_on=kSliceSchemeOn)
    return predict_densities


class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        print("Receive an TCP Request From " + str(self.client_address))

        time.sleep(1)
        print("Get An Reuquest From Client")
        last_time = time.time()
        loop_index = time_sum = time_gap = 0
        while True:
            data_len = recv_data_len(self.request)
            if not data_len:
                break
            img = recv_img(self.request, data_len)
            if not isinstance(img, np.ndarray):
                break

            print("Receive an Image")
            print("Size of Img:" + str(img.shape))

            time_gap = time.time() - last_time
            print("Time Beteen Two Img: " + str(time_gap))
            last_time = time.time()
            time_sum += time_gap
            loop_index = (loop_index + 1) % kLoop
            if not loop_index:
                print("fps: " + str(kLoop / time_sum))
                time_sum = 0

            # Compute Density
            if kIsOnline:
                pool = Pool(processes=1)
                res = pool.apply_async(_MultiProcessPredict, (img,))
                pool.close()
                pool.join()

                predict_densities = res.get()
            else:
                predict_densities = DummyPredictImgsByCaffe([img])

            # debug
            print(predict_densities)

            # Send Back
            if 0 != send_density(self.request, predict_densities[0]):
                break
        print("Connection End At Clinet!")


def StartServer(addr: str, port: int):
    try:
        server = socketserver.ThreadingTCPServer((addr, port), MyTCPHandler)
        print("Server is Listening at " + str(addr) + ":" + str(port))
        server.serve_forever()
    except Exception as err:
        print(err)
        print("Server Failed to Listen at " + str(addr) + ":" + str(port))
        server.socket.close()
        return None

if __name__ == '__main__':
    StartServer('', 12345)

# def InitServer(addr: str, port: int):
#     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     try:
#         sock.bind((addr, port))
#         sock.listen(100)
#     except Exception as err:
#         print(err)
#         print("Server Failed to Listen at " + str(addr) + ":" + str(port))
#         return None
#     else:
#         print("Server is Listening at " + str(addr) + ":" + str(port))
#         return sock