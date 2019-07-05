'''
@Date: 2019-05-16 09:18:56
@Author: Xiaobo Yang
@Email: hal_42@zju.edu.cn
@Company: Zhejiang University
@LastEditors: Xiaobo Yang
@LastEditTime: 2019-05-17 22:04:09
@Description: 
'''
#!/usr/bin/env python
# coding: utf-8

import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy.io
import os
import sys
import json
import math
import random
from random import shuffle
import h5py
import glob
import datetime
import re

from typing import Optional


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
            pt2d[pt[1], pt[0]] = 1.
            if gt_count > 1:
                sigma = distances[i][1]
            else:
                sigma = np.average(np.array(gt.shape))/2./2.  # case: 1 point
            density += scipy.ndimage.filters.gaussian_filter(
                pt2d, sigma, mode='constant')
        print('done.')
        densities.append(density)
    return densities


def load_gt_from_json(gt_file, gt_shape):
    gt = np.zeros(gt_shape, dtype='uint8')
    with open(gt_file, 'r') as jf:
        for j, dot in enumerate(json.load(jf)):
            try:
                # Xb: Attention, x, y was swaped.
                gt[int(math.floor(dot['y'])), int(math.floor(dot['x']))] = 1
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
        if os.path.isfile(gt_file.replace('.json', '.png')):
            img = cv2.imread(gt_file.replace('.json', '.png'))
            img_paths.append(gt_file.replace('.json', '.png'))
        else:
            img = cv2.imread(gt_file.replace('.json', '.jpg'))
            img_paths.append(gt_file.replace('.json', '.jpg'))
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
    return (images, gts, densities, img_paths)


"""By Xiaobo
Resize:
After resize, the density of image will be diluted of condensed.
"""


def density_resize(density, fx, fy):
    return cv2.resize(density, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)/(fx*fy)


"""By Xiaobo:
This funciton would make preparation work of parting the origin
image into several slice. In order to make sure that all slices
could have the same size, the image need to be resized so it's 
size could be devisable to the slice size.
"""


def adapt_images_and_densities(images, gts, slice_w, slice_h):
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
        out_images.append(cv2.resize(img, None, fx=fx, fy=fy,
                                     interpolation=cv2.INTER_CUBIC))
        assert out_images[-1].shape[0] % slice_h == 0 and out_images[-1].shape[1] % slice_w == 0
        if gts is not None:
            out_gts.append(density_resize(gts[i], fx, fy))
    return (out_images, out_gts)

# Generate slices


def generate_slices(images, gts, slice_w, slice_h, offset=None):
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


def _image_process(img, mean):
    img = img.copy()
    img = img.transpose(2, 0, 1).astype(np.float32)
    img -= mean
    return img


def batch_image_process(images, mean=0):
    batch = np.zeros(
        (len(images),)+images[0].transpose(2, 0, 1).shape, dtype=np.float32)
    for i, img in enumerate(images):
        batch[i] = _image_process(img, mean)
    return batch


def multiscale_pyramidal(images, gts, start=0.5, end=1.3, step=0.1):
    frange = np.arange(start, end, step)
    out_images = []
    out_gts = []
    for i, img in enumerate(images):
        for f in frange:
            out_images.append(cv2.resize(
                img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC))
            out_gts.append(density_resize(gts[i], fx=f, fy=f))
    return (out_images, out_gts)

# Data augmentation: CROP


def crop_slices(images, gts, patch_h, patch_w):
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
        p_y1, p_y2 = int((img_h - patch_h) /
                         2), int((img_h - patch_h) / 2) + patch_h
        p_x1, p_x2 = int((img_w - patch_w) /
                         2), int((img_w - patch_w) / 2) + patch_w
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
            restored_img[top:(top+img_slices_h), left:(left +
                                                       img_slices_w)] = img_slices[slices_j].copy()
            slices_j += 1
            left += img_slices_h
        left = 0
        top += img_slices_h
    return 0

def MinMaxNormalize(arr, alpha, beta):
    max = np.max(arr)
    min = np.min(arr)
    if(max - min == 0):
        return np.zeros(arr.shape, dtype='f4')
    else:
        return (arr - min) * ((beta - alpha) / (max - min)) + alpha

def AdaptImgForCaffeImgScheme(raw_imgs):
    adapted_imgs = []
    for raw_img in raw_imgs:
        flag = 0
        adapted_h = raw_img.shape[0]
        adapted_w = raw_img.shape[1]

        while adapted_h * adapted_w > 750 * 1024:
            adapted_h = int(0.9 * adapted_h)
            adapted_w = int(0.9 * adapted_w)
            flag = 1

        if not (raw_img.shape[0] % 8):
            adapted_h = adapted_h + 1
            flag = 1
        if not (raw_img.shape[1] % 8):
            adapted_w = adapted_w + 1
            flag = 1

        if not flag:
            adapted_imgs.append(raw_img.copy())
        else:
            adapted_img = np.zeros((adapted_h, adapted_w, 3), dtype='f4')
            cv2.resize(raw_img, dsize=(adapted_w, adapted_h), dst=adapted_img, interpolation=cv2.INTER_CUBIC)
            adapted_imgs.append(adapted_img)
    return adapted_imgs


def ShowTestResult(predict_densities: list, raw_imgs: Optional[list] = None,
                   gt_densities: Optional[list] = None, img_paths: Optional[list] = None,
                   is_save: bool = False, path: Optional[str]=None):
    if is_save:
        if not os.path.isdir(os.path.join(path, 'test_result')):
            os.mkdir(os.path.join(path, 'test_result'))

        if not os.path.isfile(os.path.join(path, 'test_result', 'test_result.txt')):
            os.system('touch ' + os.path.join(path,
                                              'test_result', 'test_result.txt'))
        with open(os.path.join(path, 'test_result', 'test_result.txt'), 'w') as txt_f:
            now = datetime.datetime.now()
            txt_f.write("Testing Result Recorded at: " + now.strftime("%Y-%m-%d %H:%M:%S") + '\n')

    AE = []
    SE = []
    gt_count = []
    predict_count = []
    img_names = []
    for i, predict_density in enumerate(predict_densities):
        predict_shape = (predict_density.shape[1], predict_density.shape[0])
        show_predict_density = MinMaxNormalize(predict_density, 0, 255).astype(np.uint8)
        show_predict_density = cv2.applyColorMap(show_predict_density, cv2.COLORMAP_JET)
        show_img = show_predict_density

        if raw_imgs:
            # raw_imgs[i] = raw_imgs[i][:,:,0]
            raw_imgs[i] = cv2.resize(raw_imgs[i], predict_shape, interpolation=cv2.INTER_CUBIC)
            show_img = np.concatenate((raw_imgs[i], show_img), axis=0)
        if gt_densities:
            fy = predict_density.shape[0] / gt_densities[i].shape[0]
            fx = predict_density.shape[1] / gt_densities[i].shape[1]
            resized_gt_density = density_resize(gt_densities[i], fx, fy)
            show_gt_density = MinMaxNormalize(resized_gt_density, 0, 255).astype(np.uint8)
            show_gt_density = cv2.applyColorMap(show_gt_density, cv2.COLORMAP_JET)
            diff = resized_gt_density - predict_density
            diff = MinMaxNormalize(diff, 0, 255).astype(np.uint8)
            diff = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
            show_img = np.concatenate((show_img, show_gt_density, diff))
        # plt.imshow(show_img)

        if is_save:
            if img_paths:
                srch_rslt = re.search('(\d+)\.jpg', img_paths[i])
                if srch_rslt:
                    img_name = srch_rslt.group(1)
                else:
                    img_name = re.search('(\d+)\.png', img_paths[i]).group(1)
                img_names.append(img_name)
                fname = os.path.join(path, 'test_result', img_name + '_test.jpg')
            else:
                fname = os.path.join(path, 'test_result', str(i) + 'th_test.jpg')
            plt.imsave(fname, show_img)

        # Show Test Result
        if img_paths:
            print('Image at ' + img_paths[i] + ' has test result below: \n')
        else:

            print('Image' + str(i) + 'th has test result below: \n')

        predict_count.append(np.sum(predict_densities[i]))
        print('predict_count=' + str(predict_count[-1]) + '\n')

        if gt_densities:
            gt_count.append(np.sum(gt_densities[i]))
            AE.append(abs(gt_count[-1] - predict_count[-1]))
            SE.append((gt_count[-1] - predict_count[-1]) ** 2)

            print('gt_count=' + str(gt_count[-1]) + '\n')
            print('AE=' + str(AE[-1]) + '    SE=' + str(SE[-1]) + '\n')
            print('relative error=' + str(AE[-1] / gt_count) + '\n')

        # Save Test Result
        if is_save:
            with open(os.path.join(path, 'test_result', 'test_result.txt'), 'a') as txt_f:
                if img_paths:
                    txt_f.write('Image at ' + img_paths[i] + ' has test result below: \n')
                else:
                    txt_f.write('Image' + str(i) + 'th has test result below: \n')

                txt_f.write('predict_count=' + str(predict_count[-1]) + '\n')
                if gt_densities:
                    txt_f.write('gt_count=' + str(gt_count[-1]) + '\n')
                    txt_f.write('AE=' + str(AE[-1]) + '     ')
                    txt_f.write('SE=' + str(SE[-1]) + '\n')
                    txt_f.write('relative error=' + str(AE[-1] / gt_count[-1]) + '\n')
    if gt_densities:
        MAE = np.mean(AE)
        MSE = math.sqrt(np.mean(SE))
        print('MAE=' + str(MAE))
        print('MSE=' + str(MSE))
        if img_paths:
            int_img_names = [int(i) for i in img_names]
            index = 0
            tmp_g, tmp_p, tmp_a = (gt_count[index], predict_count[index], AE[index])
            for _ in range(len(predict_densities)):
                next = int_img_names[index] - 1
                tmp_g, gt_count[next] = (gt_count[next], tmp_g)
                tmp_p, predict_count = (predict_count, tmp_p)
                tmp_a, AE[next] = (AE[next], tmp_a)
                index = next
        x_range = list(range(1, len(predict_densities) + 1))
        plt.plot(x_range, gt_count, color='Green')
        plt.plot(x_range, predict_count, color='Blue')
        plt.plot(x_range, AE, color='Red')
        MAE_fig = plt.gcf()
        plt.show()

        if is_save:
            MAE_fig.savefig(os.path.join(path, 'test_result', 'MAE.jpg'))
            with open(os.path.join(path, 'test_result', 'test_result.txt'), 'a') as txt_f:
                txt_f.write('MAE=' + str(MAE) + '\n')
                txt_f.write('MSE=' + str(MSE) + '\n')

