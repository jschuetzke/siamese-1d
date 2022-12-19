# -*- coding: utf-8 -*-
"""
November 2019

Basic building blocks for convolutional models

@author: jan schuetzke (iai)
"""

import tensorflow as tf
from tensorflow.keras import layers

def conv1d(input_layer, filters, kernel_size, strides=1, padding='same',
           activation='relu', kernel_initializer='he_uniform', 
           use_bias=False, batch_norm=False, name=None):
    x = layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, 
                      use_bias=use_bias, kernel_initializer=kernel_initializer, 
                      name=name+'_conv')(input_layer)
    if batch_norm:
        x = layers.LayerNormalization(axis=-1, scale=False, name=name+'_bn')(x)
    if activation is not None:
        x = layers.Activation(activation, name=name+'_ac')(x)
    return x

def get_conv1d_block(inputs, shrink=True, filters_shrink=4, kernel=8, stride=1,
                     filters_conv=16, block_name='cb1', batch_norm=False,
                     pooling=True, mp_size=3, mp_stride=2):
    if shrink:
        inputs = conv1d(inputs, filters=filters_shrink, kernel_size=1, 
                        batch_norm=batch_norm, name=block_name+"_shrink")
        # layers.Conv1D(kernel_size = 1, filters = filters_shrink,
        #                        strides = 1, padding='same',
        #                        activation='relu', use_bias=False,
        #                        kernel_initializer='he_uniform',
        #                        name=block_name+"_shrink")(inputs)
    conv = conv1d(inputs, filters_conv, kernel, batch_norm=batch_norm,
                  strides=stride, name=block_name)
    if pooling:
        conv = layers.MaxPooling1D(strides = mp_stride, pool_size = mp_size, 
                                   padding='same', 
                                   name=block_name+"_max_pool")(conv)
    return conv