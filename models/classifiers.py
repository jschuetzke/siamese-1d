# -*- coding: utf-8 -*-
"""
@author: jan schuetzke (iai)

Definitions of different Neural Network Architecture
"""

from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model

import models.basic_blocks as blocks

def vgg16(in_dim=3250, dropout_val=0.2, classes=28, opt=optimizers.Adam(lr=0.0005),
          trainable='all'):
    input_layer = layers.Input(shape=(in_dim, 1), name="input")
    cb = blocks.get_conv1d_block(input_layer, shrink=False, kernel=5,
                                 filters_conv=6, block_name='cb1',
                                 mp_size=2, mp_stride=2)
    cb = layers.Dropout(dropout_val, name='dropout_cb1')(cb)
    cb = blocks.get_conv1d_block(cb, shrink=False, kernel=5,
                                 filters_conv=16, block_name='cb2_1',
                                 pooling=False)
    cb = blocks.get_conv1d_block(cb, shrink=False, kernel=5,
                                 filters_conv=16, block_name='cb2_2',
                                 mp_size=2, mp_stride=2)
    cb = layers.Dropout(dropout_val, name='dropout_cb2')(cb)
    cb = blocks.get_conv1d_block(cb, shrink=False, kernel=5,
                                 filters_conv=32, block_name='cb3_1',
                                 pooling=False)
    cb = blocks.get_conv1d_block(cb, shrink=False, kernel=5,
                                 filters_conv=32, block_name='cb3_2',
                                 mp_size=2, mp_stride=2)
    cb = layers.Dropout(dropout_val, name='dropout_cb3')(cb)
    cb = blocks.get_conv1d_block(cb, shrink=False, kernel=5,
                                 filters_conv=64, block_name='cb4_1',
                                 pooling=False)
    cb = blocks.get_conv1d_block(cb, shrink=False, kernel=5,
                                 filters_conv=64, block_name='cb4_2',
                                 mp_size=2, mp_stride=2)
    cb = layers.Dropout(dropout_val, name='dropout_cb4')(cb)
    out = layers.Flatten(name='flat')(cb)
    out = layers.Dense(120, activation='relu',
                       kernel_initializer='he_uniform',
                       name='dense0')(out)
    out = layers.Dense(84, activation='relu',
                       kernel_initializer='he_uniform',
                       name='dense1')(out)
    out = layers.Dense(186, activation='relu',
                       kernel_initializer='he_uniform',
                       name='dense2')(out)
    last = layers.Dense(classes, activation='softmax', 
                        name='output')(out)
    model = Model(inputs=input_layer, outputs=last)
    if trainable == 'all':
        pass
    elif trainable == 'last':
        for i in range(len(model.layers)-5):
            model.layers[i].trainable = False
    elif trainable == 'final':
        for i in range(len(model.layers)-1):
            model.layers[i].trainable = False
    else:
        raise ValueError(f'option {trainable} for trainable not recognized!')
    model.compile(optimizer=opt, loss='categorical_crossentropy', 
                    metrics=['categorical_accuracy'])
    return model