#! /usr/bin/env python
# coding=utf-8

import core.common as common
import tensorflow as tf

def darknet53(input_data, trainable):
    # Using a name scope (instead of tf.variable_scope) for organization.
    with tf.name_scope('darknet'):
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 3, 32),
                                          trainable=trainable, name='conv0')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32, 64),
                                          trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_data = common.residual_block(input_data, 64, 32, 64,
                                               trainable=trainable, name=f'residual{i}')

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 64, 128),
                                          trainable=trainable, name='conv4', downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 128, 64, 128,
                                               trainable=trainable, name=f'residual{i+1}')

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv9', downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 256, 128, 256,
                                               trainable=trainable, name=f'residual{i+3}')

        route_1 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                          trainable=trainable, name='conv26', downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 512, 256, 512,
                                               trainable=trainable, name=f'residual{i+11}')

        route_2 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                          trainable=trainable, name='conv43', downsample=True)

        for i in range(4):
            input_data = common.residual_block(input_data, 1024, 512, 1024,
                                               trainable=trainable, name=f'residual{i+19}')

    return route_1, route_2, input_data
