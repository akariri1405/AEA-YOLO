#! /usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg
import time

class YOLOVb(object):
    """Implement TensorFlow YOLOvB network here."""
    def __init__(self, input_data, trainable, input_data_clean, defog_A=None, IcA=None):
        self.trainable = trainable  # This placeholder is still used for control later (e.g. BN training)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class = len(self.classes)
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.anchors = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method = cfg.YOLO.UPSAMPLE_METHOD
        self.isp_flag = cfg.YOLO.ISP_FLAG

        try:
            self.conv_lbbox, self.conv_mbbox, self.conv_sbbox, self.recovery_loss = \
                self.__build_nework(input_data, self.isp_flag, input_data_clean, defog_A, IcA)
        except Exception as e:
            raise NotImplementedError("Cannot build up YOLOvB network!") from e

        with tf.name_scope('pred_sbbox'):
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])
        with tf.name_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])
        with tf.name_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

    def __build_nework(self, input_data, isp_flag, input_data_clean, defog_A, IcA):
        filtered_image_batch = input_data
        self.filter_params = input_data
        filter_imgs_series = []
        if isp_flag:
            with tf.name_scope('extract_parameters_2'):
                # Use tf.image.resize (TF2) instead of deprecated tf.image.resize_images.
                input_resized = tf.image.resize(input_data, [256, 256],
                                                method=tf.image.ResizeMethod.BILINEAR)
                # IMPORTANT: pass a Python boolean (True) instead of self.trainable.
                filter_features = common.extract_parameters_2(input_resized, cfg, True)
            filters = cfg.filters
            # Instantiate each filter using the current image and configuration.
            filters = [x(filtered_image_batch, cfg) for x in filters]
            filter_parameters = []
            for j, filter in enumerate(filters):
                with tf.name_scope(f'filter_{j}'):
                    print('    creating filter:', j, 'name:', str(filter.__class__), 'abbr.', filter.get_short_name())
                    print('      filter_features:', filter_features.shape)
                    # Again, pass True instead of self.trainable.
                    filtered_image_batch, filter_parameter = filter.apply(filtered_image_batch, filter_features, defog_A, IcA)
                    filter_parameters.append(filter_parameter)
                    filter_imgs_series.append(filtered_image_batch)
                    print('      output:', filtered_image_batch.shape)
            self.filter_params = filter_parameters
        recovery_loss = tf.reduce_sum(tf.pow(filtered_image_batch - input_data_clean, 2.0))
        self.image_isped = filtered_image_batch
        self.filter_imgs_series = filter_imgs_series
        input_data = filtered_image_batch

        # Backbone network: use darknet53 with a fixed trainable flag (True)
        route_1, route_2, input_data = backbone.darknet53(input_data, True)

        input_data = common.convolutional(input_data, (1, 1, 1024, 512), True, 'conv52')
        input_data = common.convolutional(input_data, (3, 3, 512, 1024), True, 'conv53')
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), True, 'conv54')
        input_data = common.convolutional(input_data, (3, 3, 512, 1024), True, 'conv55')
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), True, 'conv56')

        conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), True, name='conv_lobj_branch')
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(self.num_class + 5)),
                                          True, name='conv_lbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1, 512, 256), True, 'conv57')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)
        with tf.name_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 768, 256), True, 'conv58')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), True, 'conv59')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), True, 'conv60')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), True, 'conv61')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), True, 'conv62')

        conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512), True, name='conv_mobj_branch')
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3*(self.num_class + 5)),
                                          True, name='conv_mbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1, 256, 128), True, 'conv63')
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)
        with tf.name_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 384, 128), True, 'conv64')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), True, 'conv65')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), True, 'conv66')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), True, 'conv67')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), True, 'conv68')

        conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), True, name='conv_sobj_branch')
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3*(self.num_class + 5)),
                                          True, name='conv_sbbox', activate=False, bn=False)

        return conv_lbbox, conv_mbbox, conv_sbbox, recovery_loss

    def decode(self, conv_output, anchors, stride):
        """
        Return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
        containing (x, y, w, h, score, probability).
        """
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = len(anchors)
        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size,
                                               anchor_per_scale, 5 + self.num_class))
        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]
        # Create the grid
        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :],
                          [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        # Clip the exponent to avoid overflow
        pred_wh = (tf.exp(tf.clip_by_value(conv_raw_dwdh, -10.0, 10.0)) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def focal(self, target, actual, alpha=1, gamma=2):
        return alpha * tf.pow(tf.abs(target - actual), gamma)

    def bbox_giou(self, boxes1, boxes2):
        eps = 1e-10
        # Convert boxes from (center, wh) to (xmin, ymin, xmax, ymax)
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
        # Ensure the coordinates are in order: (xmin, ymin, xmax, ymax)
        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / (union_area + eps)
        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - (enclose_area - union_area) / (enclose_area + eps)
        return giou

    def bbox_iou(self, boxes1, boxes2):
        eps = 1e-10
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        return inter_area / (union_area + eps)

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):
        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]
        pred_xywh     = pred[:, :, :, :, 0:4]
        pred_conf     = pred[:, :, :, :, 4:5]
        label_xywh    = label[:, :, :, :, 0:4]
        respond_bbox  = label[:, :, :, :, 4:5]
        label_prob    = label[:, :, :, :, 5:]
        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)
        bbox_loss_scale = 2.0 - label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
        iou = self.bbox_iou(pred_xywh[:, :, :, :, tf.newaxis, :],
                            bboxes[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)
        conf_focal = self.focal(respond_bbox, pred_conf)
        conf_loss = conf_focal * (respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=respond_bbox, logits=conv_raw_conf) +
                                    respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=respond_bbox, logits=conv_raw_conf))
        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob,
                                                                           logits=conv_raw_prob)
        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))
        return giou_loss, conf_loss, prob_loss

    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox,
                     true_sbbox, true_mbbox, true_lbbox):
        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors=self.anchors[0], stride=self.strides[0])
        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors=self.anchors[1], stride=self.strides[1])
        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors=self.anchors[2], stride=self.strides[2])
        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]
        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]
        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]
        with tf.name_scope('recovery_loss'):
            recovery_loss = self.recovery_loss
        return giou_loss, conf_loss, prob_loss, recovery_loss

