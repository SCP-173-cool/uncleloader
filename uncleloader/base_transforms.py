#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 22:12:56 2018

@author: loktar
"""


import sys
sys.dont_write_bytecode = True


import cv2
import random
import numpy as np

# padding modes in opencv
paddings = {'z': cv2.BORDER_CONSTANT,
            'r': cv2.BORDER_REFLECT_101}

# interpolation modes in opencv
interpolations = {'bilinear': cv2.INTER_LINEAR,
                  'bicubic': cv2.INTER_CUBIC,
                  'nearest': cv2.INTER_NEAREST}

def _apply_perspective(img, M, shape, interp_mode='bilinear', padding_mode='r'):
    """ Apply perspective transformation matrix to img with opencv API
     
    Parameters
    ----------
    img: 2-D image,
        the shape is (height, width, channel).
    M: transform matrix,
        np.array (3, 3)
        ref: https://upload.wikimedia.org/wikipedia/commons/2/2c/2D_affine_transformation_matrix.svg
     """
    return cv2.warpPerspective(img, M, shape, 
                                flags=interpolations[interp_mode], 
                                borderMode=paddings[padding_mode])

class ImageOnly(object):
    """ Apply one transform only to images
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, mask=None):
        return self.transform(img), mask


class DualCompose(object):
    """ Apply a list of transforms to data
    """
    def __init__(self, transforms, shuffle=False, ImageOnly=False):
        self.transforms = transforms
        self.ImageOnly = ImageOnly
        if shuffle:
            random.shuffle(self.transforms)

    def __call__(self, img, mask=None):
        for t in self.transforms:
            if self.ImageOnly:
                img = t(img)
            else:
                img, mask = t(img, mask)
        return img, mask


class RandomCompose(object):
    """ Randomly select some tranforms in list to apply to data.
    """
    def __init__(self, transforms, max_num=1):
        self.transforms = transforms
        self.max_num = max_num

    def __call__(self, img, mask=None):
        self.transforms = random.sample(self.transforms, self.max_num)
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

"""
PREPROCESSING FUNCTIONS
"""

class ImageResize(object):
    def __init__(self, size, interp_mode='bilinear'):
        self.size = size
        self.interp_mode = interp_mode
    
    def __call__(self, img, mask=None):
        img = cv2.resize(img, self.size, 
                         interpolation=interpolations[self.interp_mode])
        if mask is not None:
            mask = cv2.resize(mask, self.size, 
                         interpolation=interpolations[self.interp_mode])
        return img, mask

"""
AUGUMENTATION FUNCTIONS
"""
class random_horizontal_flip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        return img, mask

class random_vertical_flip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        return img, mask

class random_flip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)
        return img, mask

class random_transpose(object):
    def __int__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = img.transpose(1, 0, 2)
            if mask is not None:
                mask = mask.transpose(1, 0)
        return img, mask

