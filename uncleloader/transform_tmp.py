# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 18:06:36 2018

@author: loktarxiao
"""

import numpy as np
import cv2

def image_resize(img, size=(150, 150)):
    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

def transform(img, label):
    img = image_resize(img)
    return img, label