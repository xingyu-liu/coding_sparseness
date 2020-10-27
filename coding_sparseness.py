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
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from dnnbrain.io.fileio import ActivationFile
import statsmodels.formula.api as smf
import time
from PIL import Image

#%% specify custom paremeters
#root = os.getcwd() # path to save extracted activation
root = '/nfs/a1/userhome/liuxingyu/workingdir/coding_sparseness' # path to save extracted activation
net = 'vgg11' + '' # DNN model + ablation: ['alexnet', 'vgg11'] + ['', _permut_weight', '_permut_bias', '_norelu]

# prepare parameters
net_dir = os.path.join(root, net)
caltech256_label = pd.read_csv(os.path.join(root, 'caltech256_label'), sep='\t')

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
    
dataset = 'caltech256'  # 'imagenet', 'caltech143', 'caltech256'
act_method = 'relu'  # the sublayer to get activation from: ['relu', 'conv']
bins = 20 # bins for activation histogram

# read dnn actiation
if dataset == 'imagenet':
    stim_per_cat = 50
elif dataset == 'caltech256' or dataset == 'caltech143':
    stim_per_cat = 80

if dataset == 'caltech143':
    dnnact_path = os.path.join(
            net_dir, 'dnn_activation', '{0}_{1}_mean_{2}.act.h5'.format(
                    net, act_method, 'caltech256'))
else:
    dnnact_path = os.path.join(
            net_dir, 'dnn_activation', '{0}_{1}_mean_{2}.act.h5'.format(
                    net, act_method, dataset))
dnnact_alllayer = ActivationFile(dnnact_path).read()

if net.split('_')[0] == 'resnet152':
    layer_name = sorted(list(dnnact_alllayer.keys()), key=lambda info: (info[5], int(info[17:])))
else:
    layer_name = list(dnnact_alllayer.keys())
    
# compute PSI
sp = []
sp_img = []
sp_bincount = []
pdf_bin = []
for layer in layer_name:

    dnnact = Dnn_act(dnnact_alllayer[layer], stim_per_cat=stim_per_cat)
    sp_img.append(sparseness(dnnact.data[:,:,0].T, type='s', norm=True))
    dnnact_catmean = dnnact.cat_mean_act()[0][:, :, 0]
    
    if dataset == 'caltech143':
        dnnact_catmean = dnnact_catmean[caltech256_label['imagenet1000'] == '0', :]
    
#    if net.split('_')[-1] == 'norelu':
#        dnnact_catmean = np.abs(stats.zscore(dnnact_catmean, 0))

    dnnact_catmean_z = np.nan_to_num(stats.zscore(dnnact_catmean, 0))

    # population sparseness
    sparse_p = sparseness(dnnact_catmean_z.T, type='s', norm=True)
    sp_bincount.append(
            pd.cut(sparse_p, np.linspace(0, 1, bins+1)).value_counts().values
            /dnnact_catmean.shape[0] * 100)

    sp.append(np.squeeze(sparse_p))    
    print('{0} done'.format(layer))
    
    # pdf
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    dnnact_catmean_z_norm = min_max_scaler.fit_transform(dnnact_catmean_z.T)
    
    dist_bin = [np.histogram(dnnact_catmean_z_norm[:,i], bins=np.arange(0,1,0.01),density=True)[0] for i in range(dnnact_catmean_z_norm.shape[-1])]
    pdf_bin.append(np.asarray(dist_bin).mean(0))

sp_bincount = np.asarray(sp_bincount).T
pdf_bin = np.asarray(pdf_bin).T
sp_median = np.array([np.nanmedian(sp[i]) for i in range(len(sp))])
sp_range = [sp[i].max()-sp[i].min() for i in range(len(sp))]

# fit pdf
dist_model = ['norm','weibull']
log_lik = np.zeros((len(layer_name), len(dist_model)))
weib_paras = []
for i in range(len(layer_name)):
    data = pdf_bin[:, i] 
    row = 0
    # norm
    norm_para = stats.norm.fit(data)
    log_lik[i, row] = np.sum(stats.norm.logpdf(data, *norm_para)) 
    row += 1
    # weibull
    weib_para = stats.weibull_min.fit(data)
    log_lik[i, row] = np.sum(stats.weibull_min.logpdf(data, *weib_para)) 
    weib_paras.append(weib_para)
    row += 1
weib_k = np.asarray(weib_paras)[:,0]

# stats trend test
sp_alllayer = np.asarray(sp).reshape(-1)
h_index = np.repeat(np.arange(len(sp))+1 , sp[0].shape)
tau = stats.kendalltau(h_index, sp_alllayer)

# %% ===== 2 =====
# multi-channel category classification analysis
# ==========

net = 'vgg11'  # 'alexnet', 'vgg11'
dataset = 'caltech256'  
act_method = 'relu'  # the sublayer to get activation from: ['relu', 'conv']

# prepare parameters
stim_per_cat = 80
n_cat = 256
model_method = 'lr' 
cvfold = 2
max_iter = 10000

dnnact_path = os.path.join(
        net_dir, 'dnn_activation', '{0}_{1}_mean_{2}.act.h5'.format(
                net, act_method, dataset))
pred_dir = os.path.join(net_dir, 'dnn_prediction')
if os.path.exists(pred_dir) is False:
    os.makedirs(pred_dir)


# ----- estimating classification performance -----
dnnact_alllayer = ActivationFile(dnnact_path).read()
dnnact = np.squeeze(dnnact_alllayer['fc2_relu'])

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
                net, act_method, 'fc2', dataset, model_method))
np.save(confus_path, confus)


# ----- performance & PSI relationship -----
confus_path = os.path.join(
        pred_dir, '{0}_{1}_mean_{2}_{3}_multipred_{4}_confus.npy'.format(
                net, act_method, 'fc2', dataset, model_method))
confus = np.load(confus_path).mean(0)
acc = confus.diagonal()/(confus[0, :].sum())

acc_img = np.repeat(acc, stim_per_cat)

# layerwise correlation between category-wise sp and performance of FC2
acc_sp_corr = [stats.pearsonr(sp[i][~np.isnan(sp[i])], 
                 acc[~np.isnan(sp[i])]) for i 
               in range(len(sp))]                            
acc_sp_corr = np.asarray(acc_sp_corr)
tau_corr = stats.kendalltau(np.arange(1, len(sp)+1), acc_sp_corr[:,0])

# layerwise correlation between image-wise sp and performance of FC2
acc_sp_img_corr = [stats.pearsonr(sp_img[i][~np.isnan(sp_img[i])], 
                 acc_img[~np.isnan(sp_img[i])]) for i 
               in range(len(sp))]                            
acc_sp_img_corr = np.asarray(acc_sp_img_corr)

# ----- scatter plot of performance & PSI -----
x, y = sp, acc 

cmap = plt.cm.get_cmap('Blues')
color_norm = plt.Normalize(0, len(x)+len(x))
color_conv = cmap(color_norm(range(len(x)+len(x))))[2-len(x):, :]

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
color = ['tab:blue']*(len(x)-2)
color.extend(['tab:orange']*2)

for i in range(len(x)):
    ax = plt.subplot(gs[i])
    sns.regplot(x=x[i], y=y, color=color[i], ci=None,
                scatter_kws={'s':3}, line_kws={'color':'firebrick','linewidth':1, 'linestyle':'-'})

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

#----- plot example images ----
cat_num = 5
img_per_cat = 5

stim_file = os.path.join(root, '{0}.stim.csv'.format(dataset))
stim = pd.read_csv(stim_file, skiprows=2)
stim_path = pd.read_csv(stim_file, skiprows=0, nrows=1).iloc[0,0][5:]

# classification cat viz
rank = np.argsort(acc)[::-1]  #descending

imgs = []
#cats = rank[-np.arange(cat_num) -1]
cats = rank[np.r_[-np.arange(cat_num) -1, np.arange(cat_num)]]
for cat in cats:
    img_cat = np.random.choice(stim_per_cat, img_per_cat, replace=False)
    for i in img_cat:
        img_path = os.path.join(stim_path, stim[stim['category']==cat].iloc[i,0])
        imgs.append(Image.open(img_path))

# plot
fig, axs = plt.subplots(nrows=cats.shape[0], ncols=img_per_cat,
                        subplot_kw={'xticks': [], 'yticks': [],
                                    'frame_on': False}, figsize=[6, 9])

for i, ax in enumerate(axs.flat[:len(imgs)]):
    if np.mod(i, img_per_cat) == 0:
        ax.set_anchor('W')
        ax.set_ylabel(caltech256_label['object'][cats[i//img_per_cat]], 
                      rotation=0, horizontalalignment='left', labelpad=80)
    img = ax.imshow(imgs[i])

fig.suptitle('Categories with the worst5 (first 5 row) /best5 (last 5 row) classification performance')
plt.tight_layout()

# %% ===== 3 =====
# PSI(sp) for permutated models.
# =======================

dataset = 'imagenet'  # 'imagenet', 'caltech143', 'caltech256'
act_method = 'relu'  # the sublayer to get activation from: ['relu', 'conv']
bins = 20 # bins for activation histogram
n = 10 # number of permuted models

#
stim_per_cat = 1
sp_median = []
for i in range(n):
    dnnact_path = os.path.join(
                net_dir, 'dnn_activation', '{0}_{1}_mean_{2}_{3}.act.h5'.format(
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

    sp_median.append(np.array([np.nanmedian(sp[i]) for i in range(len(sp))]))

sp_median = np.asarray(sp_median).T
sp_median_plot = sp_median.reshape(-1)

h_index = np.repeat(np.arange(sp_median.shape[-1])+1 , sp_median.shape[0])
tau = stats.kendalltau(h_index, sp_median_plot)
