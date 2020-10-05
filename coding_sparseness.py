#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:22:55 2019

@author: liuxingyu
"""

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mytool
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from dnnbrain.io.fileio import ActivationFile
import statsmodels.formula.api as smf
import time


root = '/nfs/a1/userhome/liuxingyu/workingdir/coding_sparseness'
net = 'alexnet'  
if net in ['alexnet', 'alexnet_permut', 'alexnet_permut_weight', 'alexnet_permut_weight_chn', 
           'alexnet_permut_bias', 'alexnet_norelu']:
    layer_name = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5',
               'fc1', 'fc2']
    layer_chn_num = np.array([64, 192, 384, 256, 256, 4096, 4096])
elif net in ['vgg11', 'vgg11_permut', 'vgg11_norelu']:
    layer_name = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5',
                  'conv6', 'conv7', 'conv8', 'fc1', 'fc2']
    layer_chn_num = np.array([64, 128, 256, 256, 512, 512, 512, 512,
                              4096, 4096])   

parent_dir = os.path.join(root, net)
caltech256_label_path = os.path.join(root, 'caltech256_label')
caltech256_label = mytool.utils.readtxt(caltech256_label_path, delimiter='\t',
                                        exclude_first_n_line=1)
caltech256_label = np.asarray(caltech256_label)

#%%

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

# %% ===== 1 & 3 =====
# PSI(sp) for normal and ablated models
# =======================
    
dataset = 'imagenet'  # 'imagenet', 'caltech143', 'caltech256'
act_method = 'relu_mean'  # 'relu_mean', 'relu_max' 'conv_mean', 'conv_max'
bins = 20

if dataset == 'imagenet':
    stim_per_cat = 50
elif dataset == 'caltech256' or dataset == 'caltech143':
    stim_per_cat = 80
elif dataset == 'indoor':
    stim_per_cat = 100

if dataset == 'caltech143':
    dnnact_path = os.path.join(
            parent_dir, 'dnn_activation', '{0}_{1}_{2}.act.h5'.format(
                    net, act_method, 'caltech256'))
else:
    dnnact_path = os.path.join(
            parent_dir, 'dnn_activation', '{0}_{1}_{2}.act.h5'.format(
                    net, act_method, dataset))
dnnact_alllayer = ActivationFile(dnnact_path).read()

sp = []
sparse_p_bincount = []
pdf_bin = []
for layer in list(dnnact_alllayer.keys()):

    dnnact = Dnn_act(dnnact_alllayer[layer], stim_per_cat=stim_per_cat)       
    dnnact_catmean = dnnact.cat_mean_act()[0][:, :, 0]
    
    if dataset == 'caltech143':
        dnnact_catmean = dnnact_catmean[caltech256_label[:,-1] == '0', :]
    
    if net.split('_')[-1] == 'norelu':
        dnnact_catmean = np.abs(stats.zscore(dnnact_catmean, 0))

    dnnact_catmean_z = np.nan_to_num(stats.zscore(dnnact_catmean, 0))

    # population sparseness
    sparse_p = sparseness(dnnact_catmean_z.T, type='s', norm=True)
    sparse_p_bincount.append(
            pd.cut(sparse_p, np.linspace(0, 1, bins+1)).value_counts().values
            /dnnact_catmean.shape[0] * 100)

    sp.append(np.squeeze(sparse_p))    
    print('{0} done'.format(layer))
    
    # fit pdf
    dnnact_catmean_z_norm = mytool.core.normalize(dnnact_catmean_z.T)
    dist_bin = [np.histogram(dnnact_catmean_z_norm[:,i], bins=np.arange(0,1,0.01),density=True)[0] for i in range(dnnact_catmean_z_norm.shape[-1])]
    pdf_bin.append(np.asarray(dist_bin).mean(0))


sparse_p_bincount = np.asarray(sparse_p_bincount).T
pdf_bin = np.asarray(pdf_bin).T
sp_median = mytool.core.list_stats(sp, method='nanmedian')
sp_std = mytool.core.list_stats(sp, method='nanstd')
sp_interquartile = [np.percentile(sp[i], 75, interpolation='midpoint') -
                    np.percentile(sp[i], 25, interpolation='midpoint') for i
                    in range(len(sp))]
sp_range = [sp[i].max()-sp[i].min() for i in range(len(sp))]

# stats trend test
sp_alllayer = np.asarray(sp).reshape(-1)
h_index = np.repeat(np.arange(len(sp))+1 , sp[0].shape)
tau = stats.kendalltau(h_index, sp_alllayer)

    
# %% ===== 2 =====
# multi-channel category classification analysis
# ==========

dataset = 'caltech256'  
act_method = 'relu_mean'  # 'relu_mean', 'relu_max'
stim_per_cat = 80
n_cat = 256

model_method = 'lr' 
cvfold = 2
max_iter = 10000
top_n = 10

dnnact_path = os.path.join(
        parent_dir, 'dnn_activation', '{0}_{1}_{2}.act.h5'.format(
                net, act_method, dataset))
pred_dir = os.path.join(parent_dir, 'dnn_prediction')

# estimating classification performance
dnnact_alllayer = ActivationFile(dnnact_path).read()

for layer in list(dnnact_alllayer.keys()):
    
    dnnact = np.squeeze(dnnact_alllayer[layer])
    model_path = os.path.join(
            pred_dir, '{0}_{1}_{2}_{3}_multipred_{4}_model.pkl'.format(
                    net, act_method, layer.split('_')[0], dataset, model_method))
    
    # full model accuracy
    n_samp_fold = int(dnnact.shape[0]/cvfold)
    n_samp_fold_cat = int(n_samp_fold / n_cat)
    X = dnnact.reshape(n_samp_fold, cvfold, -1)
    Y_tra = np.asarray([int(i/(stim_per_cat/cvfold*(cvfold-1))) for i
                        in range(int(n_samp_fold*(cvfold-1)))])
    Y_test = np.asarray([int(i/(stim_per_cat/cvfold)) for i
                         in range(int(n_samp_fold))])
                   
    confus = []
    time0 = time.time()
    for cv in range(cvfold):
        X_tra = np.delete(X, cv, axis=1).reshape(-1, X.shape[-1])
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                   max_iter=max_iter)
        model.fit(X_tra, Y_tra)      

        X_test = X[:, cv, :]
        Y_pred = model.predict(X_test)
        confus.append(confusion_matrix(Y_test, Y_pred))

        print('cv {0} done, time:{1}'.format(cv, time.time()-time0))
            
    confus = np.asarray(confus)  
    confus_path = os.path.join(
            pred_dir, '{0}_{1}_{2}_{3}_multipred_{4}_confus.npy'.format(
                    net, act_method, layer.split('_')[0], dataset, model_method))
    np.save(confus_path, confus)

# performance & PSI relationship
acc = []
for layer in list(dnnact_alllayer.keys()):
    confus_path = os.path.join(
            pred_dir, '{0}_{1}_{2}_{3}_multipred_{4}_confus.npy'.format(
                    net, act_method, layer.split('_')[0], dataset, model_method))
    confus = np.load(confus_path).mean(0)
    acc.append(confus.diagonal()/(confus[0, :].sum()))

acc = np.asarray(acc)

# layer sp 2 final out ut acc
acc_sp_corr = [stats.pearsonr(sp[i][~np.isnan(sp[i])], 
                 acc[-1,~np.isnan(sp[i])]) for i 
               in range(len(sp))]
                            
acc_sp_corr = np.asarray(acc_sp_corr)

tau_corr = stats.kendalltau(np.arange(1, len(sp)+1), acc_sp_corr[:,0])

# stepwise glm
def forward_selected(data, response):
    """ref: https://planspace.org/20150423-forward_selection_with_statsmodels/
    Linear model designed by forward selection.
    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response
    response: string, name of response column in data
    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

# conv
data = pd.DataFrame(np.c_[np.asarray(sp[:-2]).T, acc[-1,:]])
data.columns = [*layer_name[:-2], 'y']
data_z = data.select_dtypes(include=[np.number]).dropna().apply(
        stats.zscore)

model = forward_selected(data_z, 'y')
print(model.model.formula)
print(model.summary())

# fc
data = pd.DataFrame(np.c_[np.asarray(sp[-2:]).T, acc[-1,:]])
data.columns = [*layer_name[-2:], 'y']
data_z = data.select_dtypes(include=[np.number]).dropna().apply(
        stats.zscore)

model = forward_selected(data_z, 'y')
print(model.model.formula)
print(model.summary())


#%% --- plot ---
cmap = plt.cm.get_cmap('Blues')
color_norm = plt.Normalize(0, len(sp)+len(sp))
color_conv = cmap(color_norm(range(len(sp)+len(sp))))[2-len(sp):, :]

cmap = plt.cm.get_cmap('Oranges')
color_norm = plt.Normalize(0, 5)
color_fc = cmap(color_norm(range(5)))[-3:-1, :]

color = np.r_[color_conv, color_fc]

if net == 'alexnet':
    ymin = 0
    yticks = np.arange(0, 1.1, 0.5)
elif net == 'vgg11':
    ymin = 0.3
    yticks = np.arange(0.3, 1.1, 0.2)    

plt.figure(figsize = (10.5,4))
gs = gridspec.GridSpec(nrows=2, ncols=5, wspace=0.1, hspace=0.1)
color = ['tab:blue']*(len(sp)-2)
color.extend(['tab:orange']*2)

for i in range(len(sp)):
    ax = plt.subplot(gs[i])
    sns.regplot(x=sp[i], y=acc[-1,:], color=color[i], ci=None,
                scatter_kws={'s':5}, line_kws={'color':'firebrick','linewidth':1, 'linestyle':'-'})

    if np.mod(i,5) == 0:
        ax.set_yticks(yticks)
        ax.set_yticklabels([])
    else:
        ax.set_yticks([])

    ax.set_xticks(np.arange(0, 0.7,0.2))
    ax.set_xticklabels([])        
    ax.set_xlim(0,0.6)
    ax.set_ylim(ymin,1.01)
    ax.set_frame_on(True)  
    ax.tick_params(axis='both', colors='lightgray', width=1, 
                   labelsize=6, labelcolor='gray')
    plt.setp(ax.spines.values(), color='lightgray')
 

# %% ===== 3 =====
# PSI(sp) for permutated models.
# =======================

dataset = 'imagenet'  # 'imagenet', 'caltech143', 'caltech256'
act_method = 'relu_mean'  # 'relu_mean', 'relu_max' 'conv_mean', 'conv_max'
bins = 20
n = 10

stim_per_cat = 1

sp_median = []
for i in range(n):
    dnnact_path = os.path.join(
                parent_dir, 'dnn_activation', '{0}_{1}_{2}_{3}.act.h5'.format(
                        net, act_method, dataset, i))
    dnnact_alllayer = ActivationFile(dnnact_path).read()
    
    sp = []
    for layer in list(dnnact_alllayer.keys()):
    
        dnnact = Dnn_act(dnnact_alllayer[layer], stim_per_cat=stim_per_cat)       
        dnnact_catmean = dnnact.cat_mean_act()[0][:, :, 0]
        
        dnnact_catmean_z = np.nan_to_num(stats.zscore(dnnact_catmean, 0))
    
        # population sparseness
        sparse_p = sparseness(dnnact_catmean_z.T, type='s', norm=True)

        sp.append(np.squeeze(sparse_p))    
        print('{0} done'.format(layer))

    sp_median.append(mytool.core.list_stats(sp, method='nanmedian'))

sp_median = np.asarray(sp_median).T
sp_median_plot = sp_median.reshape(-1)

h_index = np.repeat(np.arange(sp_median.shape[-1])+1 , sp_median.shape[0])
tau = stats.kendalltau(h_index, sp_median_plot)
