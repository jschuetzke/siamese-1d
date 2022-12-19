# -*- coding: utf-8 -*-
"""
November 2019

Siamese Networks

@author: jan schuetzke (iai)
"""
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, metrics
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from tensorflow_addons.losses import TripletHardLoss, TripletSemiHardLoss
import models.basic_blocks as blocks

def triplet_vgg16(in_dim=3250, final_filters=1, dense_output=False, 
                  dropout_val=.2, dense_features=1024, normalize=True, 
                  batch_norm=False, flatten=True):
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
    out = layers.Conv1D(kernel_size = 1, filters = final_filters,
                        strides = 1, padding='same',
                        activation='relu',
                        kernel_initializer='he_uniform',
                        name="final_shrink")(cb)
    if flatten:
        out = layers.Flatten(name='flat')(out)
    #out = layers.Lambda(lambda x: tf.squeeze(x, [-1]), name='flat')(final_shrink)
    model_name = 'triplet_vgg16'
    if batch_norm:
        model_name += '_bn'
    if dense_output:
        out = layers.Dense(dense_features, activation='relu',
                           kernel_initializer='he_uniform',
                           name='dense1')(out)
        model_name += f'dense{dense_features}'
    if normalize:
        out = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(out)
        model_name += '_l2norm'
    
    return Model(input_layer, out, name=model_name)

def triplet_phaseid(optimizer=optimizers.Adam(lr=0.0005), model_type='triplet_vgg16',
                    loss='semihard', margin=1.0, squared=False, 
                    custom_distance=None, **kwargs):
    if model_type == 'triplet_vgg16':
        model = triplet_vgg16(**kwargs)
    else:
        raise ValueError(f'Unknown model type {model_type}!')
    if custom_distance is not None:
        dist_func = custom_distance
    else:
        dist_func = 'L2'
    if 'semihard' in loss.lower():
        model.compile(optimizer=optimizer, loss=[TripletSemiHardLoss(margin, 
                                                                     squared=squared,
                                                                     distance_metric=dist_func)])
    elif 'hard' in loss.lower():
        model.compile(optimizer=optimizer, loss=[TripletHardLoss(margin, 
                                                                 squared=squared,
                                                                 distance_metric=dist_func)])
    else:
        raise ValueError(f'Unknown loss type {loss}!')
    return model

def triplet_eval(model_type='triplet_vgg16', in_dim=6500, **kwargs):
    input_a = layers.Input(shape=(in_dim, 1), name="input_compound")
    input_b = layers.Input(shape=(in_dim, 1), name="input_phase")
    if model_type == 'triplet_vgg16':
        base_network = triplet_vgg16(**kwargs)
    else:
        raise ValueError(f'Unknown model type {model_type}!')

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    L1_layer = layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    conc = L1_layer([processed_a, processed_b])
    Norm = layers.Lambda(lambda tensor: tf.norm(tensor, axis=1,
                                                keepdims=True))
    out = Norm(conc)
    model = Model([input_a, input_b], out, name='triplet_eval')
    #model.compile(optimizer=optimizer, loss=[TripletSemiHardLoss()])
    return model
