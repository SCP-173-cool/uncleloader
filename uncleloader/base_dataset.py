#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:31:52 2018

@author: scp-173
"""


"""Dataset container."""
__all__ = ['Dataset', 'SimpleDataset', 'ArrayDataset',]

import os
import numpy as np


class Dataset(object):
    """Abstract dataset class. All datasets should have this interface.

    Subclasses need to override `__getitem__`, which returns the i-th
    element, and `__len__`, which returns the total number elements.

    .. note:: An mxnet or numpy array can be directly used as a dataset.
    """
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def transform(self, fn, lazy=True):
        """Returns a new dataset with each sample transformed by the
        transformer function `fn`.

        Parameters
        ----------
        fn : callable
            A transformer function that takes a sample as input and
            returns the transformed sample.
        lazy : bool, default True
            If False, transforms all samples at once. Otherwise,
            transforms each sample on demand. Note that if `fn`
            is stochastic, you must set lazy to True or you will
            get the same result on all epochs.

        Returns
        -------
        Dataset
            The transformed dataset.
        """
        trans = _LazyTransformDataset(self, fn)
        if lazy:
            return trans
        return SimpleDataset([i for i in trans])

    def transform_first(self, fn, lazy=True):
        """Returns a new dataset with the first element of each sample
        transformed by the transformer function `fn`.

        This is useful, for example, when you only want to transform data
        while keeping label as is.

        Parameters
        ----------
        fn : callable
            A transformer function that takes the first elemtn of a sample
            as input and returns the transformed element.
        lazy : bool, default True
            If False, transforms all samples at once. Otherwise,
            transforms each sample on demand. Note that if `fn`
            is stochastic, you must set lazy to True or you will
            get the same result on all epochs.

        Returns
        -------
        Dataset
            The transformed dataset.
        """
        def base_fn(x, *args):
            if args:
                return (fn(x),) + args
            return fn(x)
        return self.transform(base_fn, lazy)

    def _fork(self):
        """Protective operations required when launching multiprocess workers."""
        # for non file descriptor related datasets, just skip
        pass


class SimpleDataset(Dataset):
    """Simple Dataset wrapper for lists and arrays.

    Parameters
    ----------
    data : dataset-like object
        Any object that implements `len()` and `[]`.
    """
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class _LazyTransformDataset(Dataset):
    """Lazily transformed dataset."""
    def __init__(self, data, fn):
        self._data = data
        self._fn = fn

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]
        if isinstance(item, tuple):
            return self._fn(*item)
        return self._fn(item)


class ArrayDataset(Dataset):
    """A dataset that combines multiple dataset-like objects, e.g.
    Datasets, lists, arrays, etc.

    The i-th sample is defined as `(x1[i], x2[i], ...)`.

    Parameters
    ----------
    *args : one or more dataset-like objects
        The data arrays.
    """
    def __init__(self, *args):
        assert len(args) > 0, "Needs at least 1 arrays"
        self._length = len(args[0])
        self._data = []
        for i, data in enumerate(args):
            assert len(data) == self._length, \
                "All arrays must have the same length; array[0] has length %d " \
                "while array[%d] has %d." % (self._length, i+1, len(data))
            
            self._data.append(data)

    def __getitem__(self, idx):
        if len(self._data) == 1:
            return self._data[0][idx]
        else:
            return tuple(data[idx] for data in self._data)

    def __len__(self):
        return self._length

