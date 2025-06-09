#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
import os
from util_filters import *  # Ensure any needed filter functions (e.g. lrelu) are defined or imported.

# --- Rewritten helper functions using tf.keras layers ---

def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):
    """
    A TF2/Keras version of the convolutional layer.
    - filters_shape: tuple like (kernel_h, kernel_w, in_channels, out_channels)
    - downsample: if True, pad and use stride 2.
    """
    if downsample:
        pad_h = (filters_shape[0] - 2) // 2 + 1
        pad_w = (filters_shape[1] - 2) // 2 + 1
        input_data = tf.pad(input_data, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode='CONSTANT')
        strides = 2
        padding = 'valid'
    else:
        strides = 1
        padding = 'same'

    conv_layer = tf.keras.layers.Conv2D(
        filters=filters_shape[-1],
        kernel_size=(filters_shape[0], filters_shape[1]),
        strides=strides,
        padding=padding,
        use_bias=not bn,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        trainable=trainable,
        name=name
    )
    conv = conv_layer(input_data)
    if bn:
        bn_layer = tf.keras.layers.BatchNormalization(trainable=trainable, name=name + '_bn')
        conv = bn_layer(conv)
    if activate:
        conv = tf.nn.leaky_relu(conv, alpha=0.1)
    return conv

def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):
    """
    A residual block that uses two convolutional layers.
    """
    short_cut = input_data
    with tf.name_scope(name):
        conv1 = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                trainable=trainable, name=name + '_conv1')
        conv2 = convolutional(conv1, filters_shape=(3, 3, filter_num1, filter_num2),
                                trainable=trainable, name=name + '_conv2')
        residual_output = conv2 + short_cut
    return residual_output

def extract_parameters(net, cfg, trainable):
    """
    Extract filter parameters from net using a small CNN.
    """
    output_dim = cfg.num_filter_parameters
    print('extract_parameters CNN:')
    channels = cfg.base_channels
    print('    ', net.shape)
    net = convolutional(net, filters_shape=(3, 3, 3, channels), trainable=trainable,
                        name='ex_conv0', downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, channels, 2 * channels), trainable=trainable,
                        name='ex_conv1', downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, 2 * channels, 2 * channels), trainable=trainable,
                        name='ex_conv2', downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, 2 * channels, 2 * channels), trainable=trainable,
                        name='ex_conv3', downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, 2 * channels, 2 * channels), trainable=trainable,
                        name='ex_conv4', downsample=True, activate=True, bn=False)
    net = tf.reshape(net, [-1, 4096])
    dense1 = tf.keras.layers.Dense(
        cfg.fc1_size,
        activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
        name='fc1'
    )(net)
    filter_features = tf.keras.layers.Dense(
        output_dim,
        activation=None,
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
        name='fc2'
    )(dense1)
    return filter_features

def extract_parameters_2(net, cfg, trainable):
    """
    A second version of extract_parameters with a smaller channel size.
    """
    output_dim = cfg.num_filter_parameters
    print('extract_parameters_2 CNN:')
    channels = 16
    print('    ', net.shape)
    net = convolutional(net, filters_shape=(3, 3, 3, channels), trainable=trainable,
                        name='ex_conv0', downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, channels, 2 * channels), trainable=trainable,
                        name='ex_conv1', downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, 2 * channels, 2 * channels), trainable=trainable,
                        name='ex_conv2', downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, 2 * channels, 2 * channels), trainable=trainable,
                        name='ex_conv3', downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, 2 * channels, 2 * channels), trainable=trainable,
                        name='ex_conv4', downsample=True, activate=True, bn=False)
    net = tf.reshape(net, [-1, 2048])
    dense1 = tf.keras.layers.Dense(
        64,
        activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
        name='fc1'
    )(net)
    filter_features = tf.keras.layers.Dense(
        output_dim,
        activation=None,
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
        name='fc2'
    )(dense1)
    return filter_features

def route(name, previous_output, current_output):
    """
    Concatenates previous and current outputs.
    """
    with tf.name_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)
    return output

def upsample(input_data, name, method="deconv"):
    """
    Upsamples the input either by resizing (nearest neighbor) or via Conv2DTranspose.
    """
    assert method in ["resize", "deconv"]
    if method == "resize":
        with tf.name_scope(name):
            input_shape = tf.shape(input_data)
            # Use tf.image.resize with nearest neighbor method.
            output = tf.image.resize(input_data,
                                     (input_shape[1] * 2, input_shape[2] * 2),
                                     method='nearest')
    elif method == "deconv":
        filters = input_data.shape[-1]
        output = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size=2,
            strides=2,
            padding='same',
            kernel_initializer=tf.random_normal_initializer(),
            name=name
        )(input_data)
    return output
