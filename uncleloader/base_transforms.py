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

paddings = {'z': cv2.BORDER_CONSTANT,
            'r': cv2.BORDER_REFLECT_101}

interpolations = {'bilinear': cv2.INTER_LINEAR,
                  'bicubic': cv2.INTER_CUBIC,
                  'nearest': cv2.INTER_NEAREST}

def _apply_perspective(img, M, shape, interp_mode='bilinear', padding_mode='r'):
     return cv2.warpPerspective(img, M, shape, 
                                flags=interpolations[interp_mode], 
                                borderMode=paddings[padding_mode])

class ImageOnly(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, mask=None):
        return self.transform(img), mask


class DualCompose(object):
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
    def __init__(self, transforms, max_num=1):
        self.transforms = transforms
        self.max_num = max_num

    def __call__(self, img, mask=None):
        self.transforms = random.sample(self.transforms, self.max_num)
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


