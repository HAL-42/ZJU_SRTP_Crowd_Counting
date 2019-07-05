#!/usr/bin/env python
# coding: utf-8

## dcc_crownet_deploy

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy.io
from skimage.transform import downscale_local_mean
import os
import sys
import json
import math
import time
# import random
# from random import shuffle
# import pickle
import h5py
import glob

#caffe
caffe_root = os.path.expanduser('~/caffe/') # change with your install location
sys.path.insert(0, os.path.join(caffe_root, 'python'))
sys.path.insert(0, os.path.join(caffe_root, 'python/caffe/proto'))
import caffe
# import caffe_pb2

import typing

# In[2]:


#constants

#Modified for lab server
model_name = 'dcc_crowdnet'
model_path = os.path.expanduser(os.path.join('../models', model_name))
data_path = os.path.expanduser(os.path.join('../data', model_name))
weights_path = os.path.expanduser(os.path.join('../weight', model_name))

dataset_paths = ['../dataset/UCF_CC_50']

slice_w = 256
slice_h = 256

patch_w = 225
patch_h = 225

net_density_h = 28
net_density_w = 28
HAS_GPU = True
GPU_ID = 0

kOutLayer = 'conv6'
kBatchSize = 40

# In[3]:


#mean
VGG_ILSVRC_16_layers_mean = np.zeros((3, patch_h, patch_w), dtype='f4')
VGG_ILSVRC_16_layers_mean[0,:,:] = 103.939
VGG_ILSVRC_16_layers_mean[1,:,:] = 116.779
VGG_ILSVRC_16_layers_mean[2,:,:] = 123.68

# In[]
"""Xb:
Find the distance to the closet point of each point using KD-Tree.
Blur each point with Gussian Filter with sigma = min_distance. 
"""
#from scipy import stats

def gaussian_filter_density(gts):
    densities = []
    for gt in gts:
        print(gt.shape)
        density = np.zeros(gt.shape, dtype=np.float32)
        gt_count = np.count_nonzero(gt)
        if gt_count == 0:
            return density

        pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))
        leafsize = 2048
        # build kdtree
        print('build kdtree...')
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
        # query kdtree
        print('query kdtree...')
        distances, locations = tree.query(pts, k=2, eps=10.)

        print('generate density...')
        for i, pt in enumerate(pts):
            pt2d = np.zeros(gt.shape, dtype=np.float32)
            pt2d[pt[1],pt[0]] = 1.
            if gt_count > 1:
                sigma = distances[i][1]
            else:
                sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
            density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        print('done.')
        densities.append(density)
    return densities

# In[4]:


def load_gt_from_json(gt_file, gt_shape):
    gt = np.zeros(gt_shape, dtype='uint8') 
    with open(gt_file, 'r') as jf:
        for j, dot in enumerate(json.load(jf)):
            try:
                gt[int(math.floor(dot['y'])), int(math.floor(dot['x']))] = 1 #Xb: Attention, x, y was swaped.
            except IndexError:
                print(gt_file, dot['y'], dot['x'], sys.exc_info())
    return gt


def load_images_and_gts(path):
    images = []
    gts = []
    densities = []
    img_paths = []
    for gt_file in glob.glob(os.path.join(path, '*.json')):
        print(gt_file)
        if os.path.isfile(gt_file.replace('.json','.png')):
            img = cv2.imread(gt_file.replace('.json','.png'))
            img_paths.append(gt_file.replace('.json','.png'))
        else:
            img = cv2.imread(gt_file.replace('.json','.jpg'))
            img_paths.append(gt_file.replace('.json','.jpg'))
        images.append(img)
        
        #load ground truth
        gt = load_gt_from_json(gt_file, img.shape[:-1])
        gts.append(gt)
        
        #densities
        desnity_file = gt_file.replace('.json','.h5')
        if os.path.isfile(desnity_file):
            #load density if exist
            with h5py.File(desnity_file, 'r') as hf:
                density = np.array(hf.get('density'))
        else:
            density = gaussian_filter_density([gt])[0]   #Xb: Blur!
            with h5py.File(desnity_file, 'w') as hf:
                hf['density'] = density
        densities.append(density)
    print(path, len(images), 'loaded')
    return (images, gts, densities, img_paths)


# In[5]:


"""By Xiaobo
Resize:
After resize, the density of image will be diluted of condensed.
"""
def density_resize(density, fx, fy):
    return cv2.resize(density, None, fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)/(fx*fy)

"""By Xiaobo:
This funciton would make preparation work of parting the origin
image into several slice. In order to make sure that all slices
could have the same size, the image need to be resized so it's 
size could be devisable to the slice size.
"""
def adapt_images_and_densities(images, gts, slice_w=slice_w, slice_h=slice_h):
    out_images = []
    out_gts = []
    for i, img in enumerate(images):
        img_h, img_w, _ = img.shape
        n_slices_h = int(round(img_h/slice_h))
        n_slices_w = int(round(img_w/slice_w))
        new_img_h = float(n_slices_h * slice_h)
        new_img_w = float(n_slices_w * slice_w)
        fx = new_img_w/img_w
        fy = new_img_h/img_h
        out_images.append(cv2.resize(img, None, fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC))
        assert out_images[-1].shape[0]%slice_h == 0 and out_images[-1].shape[1]%slice_w == 0
        if gts is not None:
            out_gts.append(density_resize(gts[i], fx, fy))
    return (out_images, out_gts)

#Generate slices
def generate_slices(images, gts, slice_w=slice_w, slice_h=slice_h, offset=None):
    if offset == None:
        offset = slice_w
    out_images = []
    out_gts = []
    for i, img in enumerate(images):
        img_h, img_w, _ = img.shape        
        p_y_id = 0
        p_y1 = 0
        p_y2 = p_y1 + slice_h
        while p_y2 <= img_h:
            p_x_id = 0
            p_x1 = 0
            p_x2 = p_x1 + slice_w
            while p_x2 <= img_w:
                out_images.append(img[p_y1:p_y2,p_x1:p_x2])
                assert out_images[-1].shape[:-1] == (slice_h, slice_w)
                if gts is not None:
                    out_gts.append(gts[i][p_y1:p_y2,p_x1:p_x2])
                    assert out_gts[-1].shape == (slice_h, slice_w)
                #next
                p_x_id += 1
                p_x1 += offset
                p_x2 += offset
            p_y_id += 1
            p_y1 += offset
            p_y2 += offset
    return (out_images, out_gts)


# In[6]:


def image_process(img, mean):
    img = img.copy()
    img = img.transpose(2, 0, 1).astype(np.float32)
    img -= mean
    return img
    
def batch_image_process(images, mean):
    batch = np.zeros((len(images),)+images[0].transpose(2, 0, 1).shape, dtype=np.float32)
    for i, img in enumerate(images):
        batch[i] = image_process(img, mean)
    return batch


# In[7]:


def predict(X_fs_deploy, mean):
    Y_fs_deploy = []
    for i, img in enumerate(X_fs_deploy):
        adapted_img, _ = adapt_images_and_densities([img], None, slice_w=patch_w, slice_h=patch_h)
        X_deploy, _ = generate_slices(adapted_img, None, slice_w=patch_w, slice_h=patch_h, offset=None)
        # net forward
        out_layer = 'conv6'
        batch_size = 10
        Y_deploy = []
        i1 = 0
        
        while i1 < len(X_deploy):
            if i1+batch_size < len(X_deploy):
                i2 = i1+batch_size
            else:
                i2 = len(X_deploy)
            batch = batch_image_process(X_deploy[i1:i2], mean)
            net.blobs['data'].reshape(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3])
            net.blobs['data'].data[...] = batch
            
            net.forward() # end=penultimate_layer
            
            for out in net.blobs[out_layer].data:
                y = out[0] #single channel
                Y_deploy.append(y)
            i1 += batch_size
        Y_fs_deploy.append(Y_deploy)
    return Y_fs_deploy


# In[8]:


model_def = os.path.join(model_path, 'deploy_addCAFFE.prototxt')


for model_weights in glob.glob(os.path.join(weights_path, 'dcc_crowdnet_train_iter_139000.caffemodel')):  #Xb: Choose the best model from train log
    
    
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    
   
    if HAS_GPU: 
        caffe.set_device(GPU_ID)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    t1 = time.time()


# In[9]:


def Restore2DImgFromSlices(restored_img, img_slices):
    img_slices_h = img_slices[0].shape[0]
    img_slices_w = img_slices[0].shape[1]
    assert restored_img.shape[0] % img_slices_h == 0 and restored_img.shape[1] % img_slices_w == 0

    h_num = restored_img.shape[0] // img_slices_h
    w_num = restored_img.shape[1] // img_slices_w
    assert len(img_slices) == h_num * w_num
    
    slices_j = 0
    top = 0
    left = 0
    for h_i in range(h_num):
        for w_i in range(w_num):
            restored_img[top:(top+img_slices_h), left:(left+img_slices_w)] = img_slices[slices_j].copy()
            slices_j += 1
            left += img_slices_h
        left = 0
        top += img_slices_h
    return 0         


# In[10]:


def PredictImgs(raw_imgs, mean):
    predict_densitys = []
    for i, raw_img in enumerate(raw_imgs):
        adapted_img, _ = adapt_images_and_densities([raw_img], None, slice_w=patch_w, slice_h=patch_h)
        img_slices, _ = generate_slices(adapted_img, None, slice_w=patch_w, slice_h=patch_h, offset=None)
        
        predict_density = np.zeros(( int(adapted_img[0].shape[0] / patch_h * net_density_h), 
                                    int(adapted_img[0].shape[1] / patch_w  * net_density_w)), dtype='f4')

        out_layer = kOutLayer
        batch_size = kBatchSize
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
        predict_densitys.append(predict_density)
    return predict_densitys


# In[11]:

def TestDataset(path: str, mean) -> int:
    raw_imgs, _, gt_densities, img_paths = load_images_and_gts(path)
    predict_densities = PredictImgs(raw_imgs, mean)

    if not os.path.isdir(os.path.join(path, 'test_result')):
        os.mkdir(os.path.join(path, 'test_result'))

    if not os.path.isfile(os.path.join(path, 'test_result', 'test_result.txt')):
        os.system('touch ' + os.path.join(path, 'test_result', 'test_result.txt'))

    with open(os.path.join(path, 'test_result', 'test_result.txt'), 'w') as txt_f:
        for i, raw_img in enumerate(raw_imgs):
            print("img" + str(i))
            show_img_line1 = np.concatenate((raw_img.transpose(2, 0, 1)[0], gt_densities[i]), axis=1)

            fx = predict_densities[i].shape[1] / gt_densities[i].shape[1]
            fy = predict_densities[i].shape[0] / gt_densities[i].shape[0]
            adapted_gt_density = density_resize(gt_densities[i], fx, fy)
            diff = predict_densities[i] - adapted_gt_density
            show_img_line2 = np.concatenate((predict_densities[i], diff), axis=1)

            show_img_line2 = np.pad(show_img_line2, ((0, 0), (0, show_img_line1.shape[1] - show_img_line2.shape[1])),
                                    'constant')
            print(show_img_line1.shape, show_img_line2.shape)

            gt_count = np.sum(gt_densities[i])
            predict_count = np.sum(predict_densities[i])
            print('gt_count=' + str(gt_count), 'predict_count' + str(predict_count))
            MAE = abs(gt_count - predict_count)
            print('MAE=' + str(MAE))
            print('relative error=' + str(MAE / gt_count))
            txt_f.write('Image at ' + img_paths[i] + ' has test result below: \n')
            txt_f.write('gt_count=' + str(gt_count), 'predict_count' + str(predict_count) + '\n')
            txt_f.write('MAE=' + str(MAE) + '     ')
            txt_f.write('relative error=' + str(MAE / gt_count) + '\n')

    return 0



if __name__ == '__main__':
    for i, dataset_path in enumerate(dataset_paths):
        print("Testing " + str(i) +"th dataset at" + dataset_path)
        TestDataset(dataset_path, VGG_ILSVRC_16_layers_mean)