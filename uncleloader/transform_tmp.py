# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 18:06:36 2018

@author: loktarxiao
"""

import numpy as np
import cv2
import random

def image_resize(img, size=(150, 150)):
    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

def resize_shorter(img, shorter_length=300):
    rows, cols = img.shape[:2]
    if rows >= cols:
        #size = (int(1.0*rows*shorter_length/cols), int(shorter_length))
        size = (int(shorter_length), int(1.0*rows*shorter_length/cols))
    else:
        #size = (int(shorter_length), int(1.0*cols*shorter_length/rows))
        size = (int(1.0*cols*shorter_length/rows), int(shorter_length))

    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

def random_flip_left_right(img):
    if random.random() > 0.5:
        return cv2.flip(img, 1)
    return img

def random_flip_up_down(img):
    if random.random() > 0.5:
        return cv2.flip(img, 0)
    return img

def random_rotate(img, rotage_range=(0, 180), random_position=False):
    angel = np.random.uniform(rotage_range[0], rotage_range[1])
    rows, cols = img.shape[:2]

    if random_position:
        cen_x = int(np.random.uniform(cols/4., 3*cols/4.))
        cen_y = int(np.random.uniform(rows/4., 3*rows/4.))
    else:
        cen_x = int(cols/2)
        cen_y = int(rows/2)
    
    M = cv2.getRotationMatrix2D((cen_x, cen_y), angel, 1)
    img = cv2.warpAffine(img, M, (rows, cols))

    return img

def random_crop(img, crop_size):
    rows, cols = img.shape[:2]
    assert rows > crop_size[0]
    assert cols > crop_size[1]

    start_x = int(np.random.uniform(0, rows - crop_size[0]))
    start_y = int(np.random.uniform(0, cols - crop_size[1]))

    end_x = int(start_x + crop_size[0])
    end_y = int(start_y + crop_size[1])

    return img[start_x:end_x, start_y:end_y, :]


def random_shear(img, range_x=(-0.5, 0.5), range_y=(0, 0)):
    rows, cols = img.shape[:2]
    shear_x = np.random.uniform(range_x[0], range_x[1])
    shear_y = np.random.uniform(range_y[0], range_y[1])
    M = np.array([1, shear_x, 0, shear_y, 1, 0]).reshape((2, 3)).astype(np.float32)
    img = cv2.warpAffine(img, M, (rows, cols))
    return img

def random_rescale(img, range_x=(0.5, 1.5), range_y=(1, 1)):
    rows, cols = img.shape[:2]
    scale_x = np.random.uniform(range_x[0], range_x[1])
    scale_y = np.random.uniform(range_y[0], range_y[1])
    M = np.array([scale_x, 0, 0, 0, scale_y, 0]).reshape((2, 3)).astype(np.float32)
    img = cv2.warpAffine(img, M, (rows, cols))
    return img


def random_hsv(img, h_range=(-720, 720), s_range=(-40, 40), v_range=(-40, 40)):
    h_add = np.random.uniform(h_range[0], h_range[1])
    s_add = np.random.uniform(s_range[0], s_range[1])
    v_add = np.random.uniform(v_range[0], v_range[1])

    img = img.copy()
    dtype = img.dtype
    if dtype != np.uint8:
        raise TypeError
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img_hsv.astype(np.int32))
    h = np.clip(abs((h + h_add) % 180), 0, 180).astype(dtype)
    s = np.clip(s + s_add, 0, 255).astype(dtype)
    v = np.clip(v + v_add, 0, 255).astype(dtype)

    img_hsv_shifted = cv2.merge((h, s, v))
    img = cv2.cvtColor(img_hsv_shifted, cv2.COLOR_HSV2RGB)
    return img


def transform(img, label):
    #img = random_rescale(img)
    #img = random_shear(img, range_x=(-0.5, 0.5), range_y=(-0.5, 0.5))
    #img = random_rotate(img, random_position=False)
    #img = image_resize(img, size=(400, 400))
    img = resize_shorter(img, shorter_length=300)
    img = random_crop(img, crop_size=(225, 225))
    img = random_hsv(img)
    
    
    #img = random_flip_left_right(img)
    return img, label