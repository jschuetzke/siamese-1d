# -*- coding: utf-8 -*-
"""
July 2019

@author: jan schuetzke (iai)
"""

import os
import numpy as np
from tensorflow.keras.utils import Sequence
import utils

from noise import simulate_noise

#%% classifier based sequences
class CompoundSequence(Sequence):

    def __init__(self, path, batch_size, scaling='log', norm='minmax', 
                 shuffle=True, shuffle_before=False):
        self.x = np.load(os.path.join(path, 'x_train.npy'))
        self.y = np.load(os.path.join(path, 'y_train.npy'))

        self.indices = np.arange(self.x.shape[0])

        if scaling is None or scaling == 'sqrt' or scaling == 'log':
            pass
        else:
            raise ValueError('Unknown scaling method!', scaling)
        self.scaling = scaling        
        self.norm = norm
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.rng = np.random.default_rng()
        if shuffle_before:
            self.on_epoch_end()
        return

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        ix = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = self.x[ix].copy()
        batch_y = self.y[ix].copy()
        
        batch_x = np.apply_along_axis(simulate_noise, 1, batch_x,
                                        2020)
        if self.scaling == 'log':
            batch_x = utils.scale_log(batch_x)
        elif self.scaling == 'sqrt':
            batch_x = utils.scale_sqrt(batch_x)
        else: # should be None
            pass
        if self.norm == 'minmax':
            batch_x = utils.scale_min_max(batch_x)
        else:
            pass # no normalziation
        return np.expand_dims(batch_x, -1), batch_y

    def on_epoch_end(self):
        if self.shuffle == True:
            self.rng.shuffle(self.indices)
        else:
            pass

#%% siamese based sequences
class TripletSeq(Sequence):

    def __init__(self, path, batch_size, log=True, norm=True, shuffle=True,
                 add_noise=True, augmentation=False, shuffle_before=False):

        self.x = np.load(os.path.join(path, 'x_train.npy'))
        self.y = np.load(os.path.join(path, 'y_train.npy'))
        self.y = np.argmax(self.y, axis=1)
        
        self.indices = np.arange(self.x.shape[0])

        self.log = log
        self.norm = norm
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.add_noise = add_noise
        self.augmentation = augmentation
        self.rng = np.random.default_rng()
        if shuffle_before:
            self.on_epoch_end()
        return

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        ix = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[ix,:].copy()
        batch_y = self.y[ix].copy()

        if self.add_noise:
            batch_x = np.apply_along_axis(simulate_noise, 1, batch_x,
                                          2020)
        if self.log:
            batch_x = utils.scale_log(batch_x)
        if self.norm:
            batch_x = utils.scale_min_max(batch_x)
        if self.augmentation:
            batch_x = np.roll(batch_x, self.rng.integers(-1000, 1000),
                                    axis=1)
        return np.expand_dims(batch_x, -1), batch_y

    def on_epoch_end(self):
        if self.shuffle == True:
            self.rng.shuffle(self.indices)
        else:
            pass