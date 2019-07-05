'''
@Date: 2019-05-16 09:17:42
@Author: Xiaobo Yang
@Email: hal_42@zju.edu.cn
@Company: Zhejiang University
@LastEditors: Xiaobo Yang
@LastEditTime: 2019-05-16 10:49:37
@Description: 
'''
#!/usr/bin/env python
# coding: utf-8

from data_preprocession import *

import numpy as np
import sys
from multiprocessing import Pool

from typing import Optional, Union


kCaffeRoot = os.path.expanduser('~/caffe/')  # change with your install location
VGG_ILSVRC_16_layers_mean = (103.939, 116.779, 123.68)


def DummyPredictImgsByCaffe(raw_imgs: Union[tuple, list]):
    random_densities = []
    for img in raw_imgs:
        random_density = np.random.rand(img.shape[0]//8, img.shape[1]//8)
        random_density.astype('f4')
        random_densities.append(random_density)
    return random_densities

try:
    sys.path.insert(0, os.path.join(kCaffeRoot, 'python'))
    sys.path.insert(0, os.path.join(kCaffeRoot, 'python/caffe/proto'))
    import caffe
    import caffe_pb2
except Exception as err:
    print("Can't Access Caffe module, if you are debugging offline, using dummy predict!")
    print(str(err))
else:
    def InitCaffeEnv(caffe_root: str, has_gpu: bool = True, gpu_id: int = 0, is_offline: bool = False) -> int:
        sys.path.insert(0, os.path.join(caffe_root, 'python'))
        sys.path.insert(0, os.path.join(caffe_root, 'python/caffe/proto'))

        if has_gpu:
            caffe.set_device(gpu_id)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        return 0


    def LoadCaffeModel(model_def: str, model_weights: str, is_offline: bool = False):
        if is_offline:
            print("Offline Debug!")
            return None
        net = caffe.Net(model_def, model_weights, caffe.TEST)
        return net


    def PredictImgsByCaffe(raw_imgs: Union[list, tuple],  net, out_layer: str,
                            patch_w: Optional=None, patch_h: Optional=None,
                            net_density_w: Optional=None, net_density_h: Optional=None,
                            batch_size=1, mean_val: tuple = VGG_ILSVRC_16_layers_mean,
                            slice_scheme_on: bool = True):
        predict_densitys = []
        if slice_scheme_on:
            mean = np.zeros((3, patch_h, patch_w), dtype='f4')
            mean[0,:,:] = mean_val[0]
            mean[1,:,:] = mean_val[1]
            mean[2,:,:] = mean_val[2]

            for i, raw_img in enumerate(raw_imgs):
                pool = Pool(processes=1)
                res = pool.apply_async(_PredictImgsByCaffe_slice,
                                (i, raw_img, mean, net, predict_densitys, out_layer,
                                patch_w, patch_h, net_density_w,net_density_h,batch_size,))
                pool.close()
                pool.join()

                predict_densitys.append(res.get())
        else:
            for raw_img in raw_imgs:
                mean = np.zeros((3, raw_img.shape[0], raw_img.shape[1]), dtype='f4')
                mean[0, :, :] = mean_val[0]
                mean[1, :, :] = mean_val[1]
                mean[2, :, :] = mean_val[2]

                pool = Pool(processes=1)
                res = pool.apply_async(_PredictImgsByCaffe_img,
                            args=(raw_img, mean, net, predict_densitys, out_layer,))
                pool.close()
                pool.join()

                predict_densitys.append(res.get())
        return predict_densitys


    def _PredictImgsByCaffe_slice(i, raw_img, mean, net, predict_densitys, out_layer: str,
                            patch_w: Optional=None, patch_h: Optional=None,
                            net_density_w: Optional=None, net_density_h: Optional=None,
                            batch_size=1):
        adapted_img, _ = adapt_images_and_densities([raw_img], None, slice_w=patch_w, slice_h=patch_h)
        img_slices, _ = generate_slices(adapted_img, None, slice_w=patch_w, slice_h=patch_h, offset=None)

        predict_density = np.zeros((int(adapted_img[0].shape[0] / patch_h * net_density_h),
                                    int(adapted_img[0].shape[1] / patch_w * net_density_w)), dtype='f4')

        density_slices = []
        i1 = 0

        while i1 < len(img_slices):
            if i1 + batch_size > len(img_slices):
                i2 = len(img_slices)
            else:
                i2 = i1 + batch_size
            batch = batch_image_process(img_slices[i1:i2], mean)
            net.blobs['data'].reshape(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3])
            net.blobs['data'].data[...] = batch

            net.forward()

            for out_slices in net.blobs[out_layer].data:
                density_slices.append(out_slices[0])

            i1 += batch_size

        Restore2DImgFromSlices(predict_density, density_slices)
        return predict_density


    def _PredictImgsByCaffe_img(raw_img, mean, net, predict_densitys, out_layer: str):
        batch = batch_image_process([raw_img], mean)
        net.blobs['data'].reshape(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3])
        net.blobs['data'].data[...] = batch
        net.forward()

        return net.blobs[out_layer][0, 0, ...]
