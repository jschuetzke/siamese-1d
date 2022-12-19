# -*- coding: utf-8 -*-
"""
@author: jan schuetzke (iai)

August 2020
Script to train VGG16-like Neural Network for phase identification for
Berlin CI Workshop Conference 2020
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecayRestarts
from tensorflow_addons.optimizers import AdamW

import utils
import models.classifiers as classifiers
from models.compound_generator import CompoundSequence

#%% load data B

x_val = np.load('./train_b/x_val_noise.npy')
x_val = utils.scale_log(x_val)
x_val = utils.scale_min_max(x_val)
y_val = np.load('./train_b/y_val.npy')

#%% prepare models
batch_size = 256
epochs = 500
cp_directory = './model_weights/'
pat = 20

#%% train
for i in range(5):
    lr_steps = utils.convert_epochs_steps(5, batch_size, 25000)
    lr_schedule = CosineDecayRestarts(0.0007, lr_steps, 2, 0.9)
    # exponential delay, 1 epoch
    wd_steps = utils.convert_epochs_steps(1, batch_size, 25000)
    wd_schedule = ExponentialDecay(0.0005, wd_steps, 0.90)
    opt = AdamW(learning_rate=lr_schedule, 
                weight_decay=lambda : None)
    opt.weight_decay = lambda : wd_schedule(opt.iterations)
    
    model = classifiers.vgg16(classes=100, opt=opt, trainable='final')
    model.load_weights(os.path.join(cp_directory, f'vgg16_c100_{i}.hdf5'))
    
    stop = EarlyStopping(patience=pat, verbose=1,
                         restore_best_weights=True, min_delta=0.0001)
    callbacks = [stop]
    
    model_name = model.name+'bt1_'+str(i)

    weights_path = os.path.join(cp_directory,(model_name+'.hdf5'))
    
    print('##############################################\n')
    print('TRAINING MODEL ', model_name)
    print('##############################################\n')

    model.fit(CompoundSequence('./train_b', batch_size,  
                               shuffle_before=True),
              epochs=epochs, verbose=2, callbacks=callbacks,
              validation_data=(x_val,y_val))
    model.save_weights(weights_path)
