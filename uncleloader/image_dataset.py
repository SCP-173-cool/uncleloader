#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:31:52 2018

@author: scp-173
"""

import sys
sys.dont_write_bytecode = True

import os
import base_dataset
import warnings
import numpy as np
from utils import image_read


class ImageFolderDataset(base_dataset.Dataset):
    
    def __init__(self, root, flag=1, transform=None):
        self._root = os.path.expanduser(root)
        self._flag = flag
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self._list_images(self._root)

    def _list_images(self, root):
        self.synsets = []
        self.items = []

        for folder in sorted(os.listdir(root)):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.'%path, stacklevel=3)
                continue
            label = len(self.synsets)
            self.synsets.append(folder)
            for filename in sorted(os.listdir(path)):
                filename = os.path.join(path, filename)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in self._exts:
                    warnings.warn('Ignoring %s of type %s. Only support %s'%(
                        filename, ext, ', '.join(self._exts)))
                    continue
                self.items.append((filename, label))

    def __getitem__(self, idx):
        img = image_read(self.items[idx][0])
        label = self.items[idx][1]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    dataset_path = '/Users/kevin/Datasets/dogs_vs_cats/train/'
    from base_transforms import *

    transform = DualCompose([ImageShorterResize(300), 
                             random_crop((224, 224)),
                             random_hsv(prob=0.8),
                             RandomCompose([
                                random_horizontal_flip(),
                                random_vertical_flip(),
                                random_flip(),
                                random_transpose(),
                                random_shear(),
                                random_rescale(),
                                random_rotate(),
                                CLAHE()
                             ], max_num=3, ImageOnly=True)], 
                    ImageOnly=True)
    print(dataset_path)
    dataset = ImageFolderDataset(dataset_path, transform=transform)

    

    for img, label in dataset:
        print(img.shape, label)
        plt.imshow(img)
        plt.show()


    