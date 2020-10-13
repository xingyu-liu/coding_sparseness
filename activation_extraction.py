#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 15:12:47 2019

@author: liuxingyu
"""

import os
import numpy as np
from dnnbrain.dnn.core import Stimulus
from dnnbrain.utils.util import gen_dmask
import torch.nn as nn
import torch

root = '/nfs/a1/userhome/liuxingyu/workingdir/coding_sparseness'
net = 'AlexNet' # 'AlexNet', 'Vgg11'
null_method = 'permut_weight_kernel'  # 'permut_weight', 'permut_weight_chn', 'permut_weight_kernel', 'permut_bias', 'norelu', 'None'
relu = True
dataset = 'imagenet'
n = 10

if dataset == 'imagenet':
    stim_per_cat = 50
elif dataset == 'caltech256' or dataset == 'caltech143':
    stim_per_cat = 80


if relu is True:
    pf = '_relu'
    sublayer = 'relu'
elif relu is False:
    pf = ''
    sublayer = 'conv'

if net == 'AlexNet':
    layer_name = ['conv1' + pf, 'conv2' + pf, 'conv3' + pf, 'conv4' + pf,
                  'conv5' + pf, 'fc1' + pf, 'fc2' + pf]
elif net == 'Vgg11':
    layer_name = ['conv1' + pf, 'conv2' + pf, 'conv3' + pf, 'conv4' + pf,
                  'conv5' + pf, 'conv6' + pf, 'conv7' + pf, 'conv8' + pf,
                  'fc1' + pf, 'fc2' + pf]

parent_dir = os.path.join(root, net.lower())
stim_path = os.path.join(root, '{0}.stim.csv'.format(dataset))

dnn = eval('db_models.{}()'.format(net))  # load DNN
stimuli = Stimulus()
stimuli.load(stim_path) # load stimuli
dmask = gen_dmask(layer_name)  # generate DNN mask

# permut weights pretrained network for each layer, finest granularity purmutation
def permut_weight_model(dnn):
    new_dict = {}
    for key in dnn.model.state_dict().keys():
        if key.split('.')[-1] == 'weight':
            para = dnn.model.state_dict()[key]
            ori_shape = para.shape
            new_dict[key] = para.view(-1)[
                    torch.randperm(*para.view(-1).size())].reshape(ori_shape)
        else:
            new_dict[key] = dnn.model.state_dict()[key]

    dnn.model.load_state_dict(new_dict)

# permut weights and bias of pretrained network for each layer, kernel-wise permuation
def permut_weight_kernel_model(dnn):
    new_dict = {}
    for key in dnn.model.state_dict().keys():
        if key.split('.')[-1] == 'weight':
            para = dnn.model.state_dict()[key]
            ori_shape = para.shape
            new_dict[key] = para.view([para.shape[0], -1])[
                    :, torch.randperm(para.view([para.shape[0], -1]).size()[-1])].reshape(ori_shape)
        else:
            new_dict[key] = dnn.model.state_dict()[key]

    dnn.model.load_state_dict(new_dict)

# permut weights pretrained network for each layer, chn-wise permuation
def permut_weight_chn_model(dnn):
    new_dict = {}
    for key in dnn.model.state_dict().keys():
        if key.split('.')[-1] == 'weight':
            para = dnn.model.state_dict()[key]
            new_dict[key] = para[torch.randperm(para.shape[0])]
        else:
            new_dict[key] = dnn.model.state_dict()[key]

    dnn.model.load_state_dict(new_dict)

# permut bias pretrained network for each layer,
def permut_bias_model(dnn):
    new_dict = {}
    for key in dnn.model.state_dict().keys():
        if key.split('.')[-1] == 'bias':
            para = dnn.model.state_dict()[key]
            ori_shape = para.shape
            new_dict[key] = para.view(-1)[
                    torch.randperm(*para.view(-1).size())].reshape(ori_shape)
        else:
            new_dict[key] = dnn.model.state_dict()[key]

    dnn.model.load_state_dict(new_dict)

# deactivate ReLU for each layer
def norelu_model(dnn):
    class Deact_ReLU(nn.Module):
        def __init__(self):
            super(Deact_ReLU, self).__init__()

        def forward(slef, x):
            return x

    def replace_relu_to_none(model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU):
                setattr(model, child_name, Deact_ReLU())
            else:
                replace_relu_to_none(child)

    replace_relu_to_none(dnn.model)


#%%
# extract activation from normal and norelu model
if null_method is None or null_method == 'norelu':

    if null_method == 'norelu':
        # prepare model
        norelu_model(dnn)
        # save path
        out_dir = os.path.join(root, '{0}_{1}'.format(net.lower(), null_method),
                                'dnn_activation')
        if os.path.exists(out_dir) is not True:
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, '{0}_{1}_{2}_mean_{3}.act.h5'.format(
                net.lower(), null_method, sublayer, dataset))#

    elif null_method is None:
        # save path
        out_dir = os.path.join(root, net.lower(),'dnn_activation')
        if os.path.exists(out_dir) is not True:
            os.makedirs(out_dir)

        out_path = os.path.join(out_dir, '{0}_{1}_mean_{2}.act.h5'.format(
                net.lower(), sublayer, dataset))

    # extract activation
    activation = dnn.compute_activation(stimuli, dmask, 'mean', cuda=True)
    # save act
    activation.save(out_path)


#%%
# extract activation from weight permuted model
if null_method in ['permut_weight', 'permut_weight_chn', 'permut_weight_kernel', 'permut_bias']:
    for i in range(n):

        # prepare model
        if null_method == 'permut_weight':
            permut_weight_model(dnn)
        elif null_method == 'permut_weight_chn':
            permut_weight_chn_model(dnn)
        elif null_method == 'permut_weight_kernel':
            permut_weight_kernel_model(dnn)
        elif null_method == 'permut_bias':
            permut_bias_model(dnn)

        # extract activation-
        activation = dnn.compute_activation(stimuli, dmask, 'mean', cuda=True)

        # compute catwise mean and replace the original stimuliwise act
        for layer in list(activation.layers):
            cur_shape = np.shape(activation.get(layer))
            new_shape = np.r_[int(cur_shape[0] / stim_per_cat),
                              stim_per_cat, cur_shape[1:]]
            cat_data = activation.get(layer).reshape(new_shape)
            activation.set(layer, cat_data.mean(1))

        # save act
        out_dir = os.path.join(root, '{0}_{1}'.format(net.lower(), null_method),
                                'dnn_activation')
        if os.path.exists(out_dir) is not True:
            os.makedirs(out_dir)

        out_path = os.path.join(out_dir, '{0}_{1}_{2}_mean_{3}_{4}.act.h5'.format(
                net.lower(), null_method, sublayer, dataset, i))
        activation.save(out_path)
