#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: dcc_crownet.py
@time: 2019/5/13 11:52
@desc:
"""

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
import random
from random import shuffle
import pickle
import h5py
import glob

model_name = 'dcc_crowdnet'
model_path = os.path.expanduser(os.path.join('..\\models', model_name))
data_path = os.path.expanduser(os.path.join('..\\data', model_name))
weights_path = os.path.expanduser(os.path.join('..\\weight', model_name))

dataset_paths = ['../dataset/UCF_CC_50']

slice_w = 256
slice_h = 256

patch_w = 225
patch_h = 225

net_density_h = 28
net_density_w = 28

#mean
VGG_ILSVRC_16_layers_mean = np.zeros((3, patch_h, patch_w), dtype='f4')
VGG_ILSVRC_16_layers_mean[0,:,:] = 103.939
VGG_ILSVRC_16_layers_mean[1,:,:] = 116.779
VGG_ILSVRC_16_layers_mean[2,:,:] = 123.68


def load_gt_from_json(gt_file, gt_shape):
    gt = np.zeros(gt_shape, dtype='uint8')
    with open(gt_file, 'r') as jf:
        for j, dot in enumerate(json.load(jf)):
            try:
                gt[int(math.floor(dot['y'])), int(math.floor(dot['x']))] = 1  # Xb: Attention, x, y was swaped.
            except IndexError:
                print(gt_file, dot['y'], dot['x'], sys.exc_info())
    return gt


def load_images_and_gts(path):
    images = []
    gts = []
    densities = []
    for gt_file in glob.glob(os.path.join(path, '*.json')):
        print(gt_file)
        if os.path.isfile(gt_file.replace('.json', '.png')):
            img = cv2.imread(gt_file.replace('.json', '.png'))
        else:
            img = cv2.imread(gt_file.replace('.json', '.jpg'))
        images.append(img)

        # load ground truth
        gt = load_gt_from_json(gt_file, img.shape[:-1])
        gts.append(gt)

        # densities
        desnity_file = gt_file.replace('.json', '.h5')
        if os.path.isfile(desnity_file):
            # load density if exist
            with h5py.File(desnity_file, 'r') as hf:
                density = np.array(hf.get('density'))
        else:
            density = gaussian_filter_density([gt])[0]  # Xb: Blur!
            with h5py.File(desnity_file, 'w') as hf:
                hf['density'] = density
        densities.append(density)
    print(path, len(images), 'loaded')
    return (images, gts, densities)


"""By Xiaobo
Resize:
After resize, the density of image will be diluted of condensed.
"""


def density_resize(density, fx, fy):
    return cv2.resize(density, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC) / (fx * fy)


def multiscale_pyramidal(images, gts, start=0.5, end=1.3, step=0.1):
    frange = np.arange(start, end, step)
    out_images = []
    out_gts = []
    for i, img in enumerate(images):
        for f in frange:
            out_images.append(cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC))
            out_gts.append(density_resize(gts[i], fx=f, fy=f))
    return (out_images, out_gts)


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
        n_slices_h = int(round(img_h / slice_h))
        n_slices_w = int(round(img_w / slice_w))
        new_img_h = float(n_slices_h * slice_h)
        new_img_w = float(n_slices_w * slice_w)
        fx = new_img_w / img_w
        fy = new_img_h / img_h
        out_images.append(cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC))
        assert out_images[-1].shape[0] % slice_h == 0 and out_images[-1].shape[1] % slice_w == 0
        if gts is not None:
            out_gts.append(density_resize(gts[i], fx, fy))
    return (out_images, out_gts)


# Generate slices
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
                out_images.append(img[p_y1:p_y2, p_x1:p_x2])
                assert out_images[-1].shape[:-1] == (slice_h, slice_w)
                if gts is not None:
                    out_gts.append(gts[i][p_y1:p_y2, p_x1:p_x2])
                    assert out_gts[-1].shape == (slice_h, slice_w)
                # next
                p_x_id += 1
                p_x1 += offset
                p_x2 += offset
            p_y_id += 1
            p_y1 += offset
            p_y2 += offset
    return (out_images, out_gts)


# Data augmentation: CROP
def crop_slices(images, gts):
    out_images = []
    out_gts = []
    for i, img in enumerate(images):
        # data augmentation
        # crop-5
        img_h, img_w, _ = img.shape
        gt = gts[i]
        # top-left
        p_y1, p_y2 = 0, patch_h
        p_x1, p_x2 = 0, patch_w
        out_images.append(img[p_y1:p_y2, p_x1:p_x2])
        out_gts.append(gt[p_y1:p_y2, p_x1:p_x2])
        # top-right
        p_y1, p_y2 = 0, patch_h
        p_x1, p_x2 = img_w - patch_w, img_w
        out_images.append(img[p_y1:p_y2, p_x1:p_x2])
        out_gts.append(gt[p_y1:p_y2, p_x1:p_x2])
        # bottom-left
        p_y1, p_y2 = img_h - patch_h, img_h
        p_x1, p_x2 = 0, patch_w
        out_images.append(img[p_y1:p_y2, p_x1:p_x2])
        out_gts.append(gt[p_y1:p_y2, p_x1:p_x2])
        # bottom-right
        p_y1, p_y2 = img_h - patch_h, img_h
        p_x1, p_x2 = img_w - patch_w, img_w
        out_images.append(img[p_y1:p_y2, p_x1:p_x2])
        out_gts.append(gt[p_y1:p_y2, p_x1:p_x2])
        # center
        p_y1, p_y2 = int((img_h - patch_h) / 2), int((img_h - patch_h) / 2) + patch_h
        p_x1, p_x2 = int((img_w - patch_w) / 2), int((img_w - patch_w) / 2) + patch_w
        out_images.append(img[p_y1:p_y2, p_x1:p_x2])
        out_gts.append(gt[p_y1:p_y2, p_x1:p_x2])
    return (out_images, out_gts)


"""Xb:
Implement with crop.
10 Crop
"""


# Data augmentation: FLIP
def flip_slices(images, gts):
    out_images = []
    out_gts = []
    for i, img in enumerate(images):
        img_h, img_w, _ = img.shape
        gt = gts[i]
        # original
        out_images.append(img)
        out_gts.append(gt)
        # flip: left-right
        out_images.append(np.fliplr(img))
        out_gts.append(np.fliplr(gt))
    return (out_images, out_gts)


# Shuffling
def shuffle_slices(images, gts):
    out_images = []
    out_gts = []
    index_shuf = list(range(len(images)))
    shuffle(index_shuf)
    for i in index_shuf:
        out_images.append(images[i])
        out_gts.append(gts[i])
    return (out_images, out_gts)


"""Xb:
正类：多于一个人  负类：少于一个人
正类里随机抽样，人越多越可能抽到
目标负类数目是正类1/6，从负类里随机抽样
"""


def samples_distribution(images, gts):
    out_images = []
    out_gts = []
    gts_count = list(map(np.sum, gts))
    max_count = max(gts_count)
    # pos
    for i, img in enumerate(images):
        if gts_count[i] >= 1. and random.random() < gts_count[i] ** 2 / max_count ** 2:
            out_images.append(img)
            out_gts.append(gts[i])
    # neg
    neg_count = sum(gt_count < 1. for gt_count in gts_count)
    obj_neg_count = len(out_gts) / 6  # ~= 15-16%
    neg_keep_prob = min(1., float(obj_neg_count) / float(neg_count))
    for i, img in enumerate(images):
        if gts_count[i] < 1. and random.random() < neg_keep_prob:
            out_images.append(img)
            out_gts.append(gts[i])

    return (out_images, out_gts)

"""Xb:
Find the distance to the closet point of each point using KD-Tree.
Blur each point with Gussian Filter with sigma = min_distance. 
"""
from scipy import stats

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


def image_process(img, mean):
    img = img.copy()
    img = img.transpose(2, 0, 1).astype(np.float32)
    img -= mean
    return img


def batch_image_process(images, mean):
    batch = np.zeros((len(images),) + images[0].transpose(2, 0, 1).shape, dtype=np.float32)
    for i, img in enumerate(images):
        batch[i] = image_process(img, mean)
    return batch




# Positive image and ground truth loading
X_fs = []
Y_fs = []

# ----------------------------MAIN---------------------------------
if __name__ == '__main__':
    # Positive image and ground truth loading
    X_fs = []
    Y_fs = []

    for path in dataset_paths:
        images, gts, densities = load_images_and_gts(path)
        X_fs += images
        Y_fs += densities
