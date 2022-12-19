# -*- coding: utf-8 -*-
"""
@author: jan schuetzke (iai)
"""
import os
import re
import math
import numpy as np
from tqdm import tqdm


def pair(directory, datapoints=2500, shuffle=False, ending='.data', log=True,
         normalize=True):
    classes = os.path.join(directory, 'compound_info.csv')

    y = np.loadtxt(classes, delimiter=',', skiprows=1)

    x_files = [os.path.join(directory,f) for f in os.listdir(directory)\
               if ending in f]
    # empty array for values
    x = np.zeros([len(x_files),datapoints])
    for file in tqdm(x_files):
        if ending == '.data':
            temp = np.loadtxt(file)
        else:
            temp = np.loadtxt(file, usecols=1)
        if log:
            temp[temp < 1] = 1
            temp = np.log10(temp)
        if normalize:
            temp = scale_min_max(temp)
        number = int(re.match('.*[/\\\](n_)?compound_([\d]*)\.[a-z]+',
                              file).group(2))
        x[number,:] = temp

    if shuffle:
        x, y = shuffle_arrays(x, y)

    x = np.expand_dims(x, -1)
    y[y > 0] = 1
    return x,y

def npy(directory, shuffle=False, scale='log', normalize=True, half=False,
        mmap=None, expand=True, noise=True, binarize=True):
    if noise:
        filename = 'x_noise.npy'
    else:
        filename = 'x.npy'
    if mmap is None:
        x = np.load(os.path.join(directory, filename))
    else:
        x = np.load(os.path.join(directory, filename), mmap_mode=mmap)
    y = np.loadtxt(os.path.join(directory, 'compound_info.csv'),
                   delimiter=',', skiprows=1)
    if half:
        x = x[:,::2]
    if scale == 'log':
        x = scale_log(x)
    elif scale == 'sqrt':
        x = scale_sqrt(x)
    elif scale is None:
        pass
    else:
        raise ValueError('Unknown Scaling method!', scale)
    if normalize:
        x = scale_min_max(x)
    if shuffle:
        x, y = shuffle_arrays(x, y)
    if expand:
        x = np.expand_dims(x, -1)
    if binarize:
        y[y > 0] = 1
    return x, y

def single_dir(directory, datapoints=2500, shuffle=False, ending='.data',
               log=True, normalize=True):
    x_files = [os.path.join(directory,f) for f in os.listdir(directory)\
               if ending in f]
    # empty array for values
    x = np.zeros([len(x_files),datapoints])
    for file in tqdm(x_files):
        if ending == '.data':
            temp = np.loadtxt(file)
        else:
            temp = np.loadtxt(file, usecols=1)
        if log:
            temp[temp < 1] = 1
            temp = np.log10(temp)
        if normalize:
            temp /= np.max(np.abs(temp),axis=0)
        number = int(re.match('.*[/\\\](n_)?compound_([\d]*)\.[a-z]+',file).group(2))
        x[number,:] = temp

    if shuffle:
        np.random.shuffle(x)

    x = np.expand_dims(x, -1)
    return x

def load_data_old(directory, datapoints=2500, shuffle=True):
    classes = os.path.join(directory, 'classes.dat')
    try:
        y = np.loadtxt(classes, delimiter=',')
    except:
        y = np.loadtxt(classes, usecols=range(1,29))

    x_files = [os.path.join(directory,f) for f in os.listdir(directory)\
               if '.xye' in f]
    # empty array for values
    x = np.zeros([len(x_files),datapoints])
    for file in tqdm(x_files):
        temp = np.loadtxt(file, usecols=1)
        number = int(re.match('.*[/\\\]data_([\d]*).xye',file).group(1))
        x[(number-1),:] = temp

    if shuffle:
        new = np.c_[x,y]
        np.random.seed(32)
        np.random.shuffle(new)
        x = new[:,:datapoints]
        y = new[:,datapoints:]

    x = np.expand_dims(x, -1)
    return x,y

# =============================================================================
# def load_data_multiple(*argv):
#     x_all = []
#     y_all = []
#     for folder in argv:
#         x_temp, y_temp = load_data(folder)
#         x_all.append(x_temp)
#         y_all.append(y_temp)
#     return x_all, y_all
# =============================================================================

def resample(path, start=5.01, end=70.01, step=0.01):
    raw = np.loadtxt(path)
    return np.interp(np.arange(start, end, step), raw[:,0], raw[:,1])

#%%
def scale_log(ndarray):
    if ndarray.ndim == 1:
        return np.log10(ndarray - np.min(ndarray, axis=0, keepdims=True)+10)
    else:
        return np.log10(ndarray - np.min(ndarray, axis=1, keepdims=True)+10)

def scale_sqrt(ndarray):
    if ndarray.ndim == 1:
        return np.sqrt(ndarray - np.min(ndarray, axis=0, keepdims=True)+1)
    else:
        return np.sqrt(ndarray - np.min(ndarray, axis=1, keepdims=True)+1)

def scale_min_max(ndarray, clip_perc=False, perc=0.2, 
                  output_max=False, input_max=None):
    x = ndarray.copy()
    if clip_perc:
        if x.ndim == 1:    
            pc = np.percentile(x, perc)
            x = np.clip(x, pc, None)
        else:
            pc = np.percentile(x, perc, axis=1)
            x = np.clip(x, np.expand_dims(pc, -1), None)
    max_arr = input_max
    if x.ndim == 1:
        min_arr = np.min(x, axis=0)
        if max_arr is None:
            max_arr = np.max(x, axis=0)
    else:
        min_arr = np.min(x, axis=1, keepdims=True)
        if max_arr is None:
            max_arr = np.max(x, axis=1, keepdims=True)
    if output_max:
        return ((x - min_arr) / (max_arr - min_arr)), max_arr
    # deprecated
    # if x.ndim == 1:
    #     x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    # else:
    #     x = (x - x.min(axis=1, keepdims=True)) / \
    #         (x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True))
    return (x - min_arr) / (max_arr - min_arr)

def shuffle_arrays(x,y):
    _, div = x.shape
    new = np.c_[x,y]
    np.random.shuffle(new)
    return new[:,:div], new[:,div:]

def get_npy_size(npy_path):
    temp = np.load(npy_path, mmap_mode='r')
    return temp.shape

def convert_epochs_steps(epochs, batch_size, length):
    return math.ceil(length/batch_size) * epochs