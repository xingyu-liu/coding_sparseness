#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 15:09:35 2020

@author: liuxingyu
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Dnn_act:
    """ dnn activation

    Attibutes
    ---------

        data: array_like,
            shape = [n_stim,n_chn,n_unit_in_row,n_unit_in_column]

    """

    def __init__(self, data, stim_per_cat):

        if data.ndim == 2 or data.ndim == 3:
            self.data = data
        elif data.ndim == 4:
            self.data = data.reshape([data.shape[0], data.shape[1], -1])
        else:
            print('the shape of data has to be 2/3/4-d')
        self.stim_num = np.shape(self.data)[0]
        self.stim_per_cat = stim_per_cat
        self.cat_num = int(self.stim_num / self.stim_per_cat)
        self.chn_num = np.shape(self.data)[1]
        self.relued = False

    def relu(self):
        self.data[self.data < 0] = 0
        self.relued = True

    def chn_act(self, top_n=5, replace=False):
        unit_max_act = np.sort(self.data, -1)[:, :, -1*top_n:]
        if replace is True:
            self.data = unit_max_act.mean(-1)
        else:
            return unit_max_act.mean(-1)

    def cat_mean_act(self):
        cur_shape = np.shape(self.data)
        new_shape = np.r_[self.cat_num, self.stim_per_cat, cur_shape[1:]]
        cat_data = self.data.reshape(new_shape)
        return cat_data.mean(1), cat_data.std(1)
    

def sparseness(x, type='s', norm=False):
    """
    parameters:
    ----------
        x: [n_sitm] or [n_stim, n_cell], firing rate(activation) of each cell 
            to each stimulus
    """
    
    if np.ndim(x) == 1:
        x = x[:, np.newaxis]
        
    if norm is True:

        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        x = min_max_scaler.fit_transform(x)
   
    n_stim = x.shape[0]

    # make sure any x > 0
    assert x.min() >= 0, 'x should all be positive'
    
    sparse_v = ((x.sum(0)/n_stim)**2) / (
            np.asarray([*map(lambda x: x**2, x)]).sum(0)/n_stim)
    # set sparse_v of cells that are always silent to 1
    sparse_v[x.sum(0) == 0] = 1
    
    if type == 's':
        sparse_v = (1 - sparse_v) / (1 - 1/n_stim)

    return sparse_v