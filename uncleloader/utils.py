#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 22:12:56 2018

@author: loktar
"""

import cv2
import numpy as np

def image_read(img_path):
    """ The faster image reader with opencv API
    """
    with open(img_path, 'rb') as fp:
        raw = fp.read()
        img = cv2.imdecode(np.asarray(bytearray(raw), dtype="uint8"), cv2.IMREAD_COLOR)
        img = img[:,:,::-1]

    return img


## function test block
if __name__ == "__main__":

    """ "image read" Test block
    """
    import time
    img_path = './dark.png'
    N = 100
    tic = time.time()
    for i in range(N):
        img = image_read(img_path)
    print(N/(time.time()-tic), 'images decoded per second with mx.image')