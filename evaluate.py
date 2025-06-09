#! /usr/bin/env python
# coding=utf-8

import os
import re
import cv2
import time
import math
import random
import shutil
import numpy as np
import tensorflow as tf

# Disable eager execution for TF1-style sessions.
tf.compat.v1.disable_eager_execution()

import core.utils as utils
from core.config import cfg, args
from core.yolovb import YOLOVb  # Must match the architecture used during training
from filters import *

# Build experiment folder using Windows-friendly join
exp_folder = os.path.join(args.exp_dir, f"exp_{args.exp_num}")

# Set visible GPU devices.
if args.use_gpu == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def strip_suffix(name):
    """Strip trailing underscore-number suffix (if present)."""
    return re.sub(r'_\d+$', '', name)

# -------------------------
# Interactive Test Mode Selection
# -------------------------
def select_test_mode():
    """
    Let the user choose between quick test and complete test.
    The quick test annotation file is taken from args.test_path.
    The complete test annotation file is set here as an example.
    If the chosen file does not exist, the options are displayed again.
    """
    # Get quick test file from command-line/config (it may be a list)
    quick_test_file = args.test_path[0] if isinstance(args.test_path, list) else args.test_path
    # Set complete test file (adjust this path as needed)
    # For test VOC dataset
    complete_test_file = r"C:\Users\ENG-LT-SL-08\Desktop\AEA-YOLO\AEA-YOLO\files\voc_norm_train.txt"
    
    while True:
        print("\nSelect Test Mode:")
        print("1. Quick Test (images file: {})".format(quick_test_file))
        print("2. Complete Test (Annotation file: {})".format(complete_test_file))
        choice = input("Enter option (1 or 2): ").strip()
        if choice == "1":
            if os.path.exists(quick_test_file):
                print("Quick Test file found.")
                return quick_test_file
            else:
                print("Quick Test file not found: {}\n".format(quick_test_file))
        elif choice == "2":
            if os.path.exists(complete_test_file):
                print("Complete Test file found.")
                return complete_test_file
            else:
                print("Complete Test file not found: {}\n".format(complete_test_file))
        else:
            print("Invalid choice. Please enter 1 or 2.")

class YoloTest(object):
    def __init__(self):
        # Basic parameters
        self.input_size = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        # Use the score threshold defined in the test config.
        self.score_threshold = cfg.TEST.SCORE_THRESHOLD  
        self.iou_threshold = cfg.TEST.IOU_THRESHOLD

        # -------------------------
        # Instead of using a command-line option, prompt the user to choose the test mode.
        # -------------------------
        self.annotation_path = select_test_mode()

        # Get weight file from config
        self.weight_file = (cfg.TEST.WEIGHT_FILE[0]
                            if isinstance(cfg.TEST.WEIGHT_FILE, list)
                            else cfg.TEST.WEIGHT_FILE)
        self.write_image = cfg.TEST.WRITE_IMAGE
        self.write_image_path = (cfg.TEST.WRITE_IMAGE_PATH[0]
                                 if isinstance(cfg.TEST.WRITE_IMAGE_PATH, list)
                                 else cfg.TEST.WRITE_IMAGE_PATH)
        self.show_label = cfg.TEST.WRITE_IMAGE_SHOW_LABEL
        self.isp_flag = cfg.YOLO.ISP_FLAG

        # If the weight_file is a directory, use the latest checkpoint.
        if os.path.isdir(self.weight_file):
            latest_ckpt = tf.compat.v1.train.latest_checkpoint(self.weight_file)
            if latest_ckpt is None:
                raise ValueError("No checkpoint found in directory: " + self.weight_file)
            else:
                self.weight_file = latest_ckpt
                print("Restoring from latest checkpoint:", self.weight_file)
        else:
            if not os.path.exists(self.weight_file + ".index"):
                raise ValueError("The passed save_path is not a valid checkpoint: " + self.weight_file)

        # Define input placeholders.
        with tf.compat.v1.name_scope('input'):
            self.input_data = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_data')
            self.defog_A = tf.compat.v1.placeholder(tf.float32, [None, 3], name='defog_A')
            self.IcA = tf.compat.v1.placeholder(tf.float32, [None, None, None, 1], name='IcA')
            self.trainable = tf.compat.v1.placeholder(tf.bool, name='trainable')
            self.input_data_clean = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3],
                                                               name='input_data_clean')

        # Build the YOLOVb model.
        model = YOLOVb(self.input_data, self.trainable, self.input_data_clean, self.defog_A, self.IcA)
        (self.pred_sbbox, self.pred_mbbox, self.pred_lbbox, self.image_isped,
         self.isp_params, self.filter_imgs_series) = (model.pred_sbbox, model.pred_mbbox,
                                                     model.pred_lbbox, model.image_isped,
                                                     model.filter_params, model.filter_imgs_series)

        # Create session.
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)

        # Build custom mapping dictionary for checkpoint restoration.
        all_vars = tf.compat.v1.global_variables()
        reader = tf.compat.v1.train.NewCheckpointReader(self.weight_file)
        ckpt_keys = reader.get_variable_to_shape_map().keys()
        var_list = {}
        for var in all_vars:
            gname = var.op.name
            if gname in ckpt_keys:
                var_list[gname] = var
            else:
                alt_name = "define_loss/" + gname
                if alt_name in ckpt_keys:
                    var_list[alt_name] = var
                else:
                    stripped = strip_suffix(gname)
                    if stripped in ckpt_keys:
                        var_list[stripped] = var
                    else:
                        alt_stripped = "define_loss/" + stripped
                        if alt_stripped in ckpt_keys:
                            var_list[alt_stripped] = var

        # Restore variables using Saver.
        saver = tf.compat.v1.train.Saver(var_list=var_list)
        saver.restore(self.sess, self.weight_file)
        print("Saver.restore() complete.\n")

        # Initialize any uninitialized variables.
        uninit_names = self.sess.run(tf.compat.v1.report_uninitialized_variables())
        if uninit_names.size:
            uninit_vars = [v for v in all_vars if v.op.name.encode() in uninit_names]
            self.sess.run(tf.compat.v1.variables_initializer(uninit_vars))
        print("Checkpoint loading complete.\n")

    def predict(self, image, image_name):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape
        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]  # Add batch dimension

        def DarkChannel(im):
            b, g, r = cv2.split(im)
            return cv2.min(cv2.min(r, g), b)

        def AtmLight(im, dark):
            h, w = im.shape[:2]
            imsz = h * w
            numpx = int(max(math.floor(imsz / 1000), 1))
            darkvec = dark.reshape(imsz, 1)
            imvec = im.reshape(imsz, 3)
            indices = darkvec.argsort(0)
            indices = indices[(imsz - numpx):imsz]
            atmsum = np.zeros([1, 3])
            for ind in range(1, numpx):
                atmsum += imvec[indices[ind]]
            return atmsum / numpx

        def DarkIcA(im, A):
            im3 = np.empty(im.shape, im.dtype)
            for ind in range(0, 3):
                im3[:, :, ind] = im[:, :, ind] / A[0, ind]
            return DarkChannel(im3)

        if self.isp_flag:
            dark = np.zeros((image_data.shape[0], image_data.shape[1], image_data.shape[2]))
            defog_A = np.zeros((image_data.shape[0], image_data.shape[3]))
            IcA = np.zeros((image_data.shape[0], image_data.shape[1], image_data.shape[2]))
            if any('DefogFilter' in str(f) for f in cfg.filters):
                for i in range(image_data.shape[0]):
                    dark_i = DarkChannel(image_data[i])
                    defog_A_i = AtmLight(image_data[i], dark_i)
                    IcA_i = DarkIcA(image_data[i], defog_A_i)
                    dark[i, ...] = dark_i
                    defog_A[i, ...] = defog_A_i
                    IcA[i, ...] = IcA_i
            IcA = np.expand_dims(IcA, axis=-1)
            start_time = time.time()
            (pred_sbbox, pred_mbbox, pred_lbbox, image_isped, isp_param, filter_imgs_series) = \
                self.sess.run([self.pred_sbbox, self.pred_mbbox, self.pred_lbbox, self.image_isped,
                               self.isp_params, self.filter_imgs_series],
                              feed_dict={
                                  self.input_data: image_data,
                                  self.defog_A: defog_A,
                                  self.IcA: IcA,
                                  self.trainable: False,
                                  self.input_data_clean: image_data
                              })
            time_one_img = time.time() - start_time
            print('Processing one image took: {:.2f} sec'.format(time_one_img))
        else:
            start_time = time.time()
            (pred_sbbox, pred_mbbox, pred_lbbox, image_isped, isp_param) = \
                self.sess.run([self.pred_sbbox, self.pred_mbbox, self.pred_lbbox, self.image_isped, self.isp_params],
                              feed_dict={self.input_data: image_data, self.trainable: False})
            time_one_img = time.time() - start_time
            print('Processing one image took: {:.2f} sec'.format(time_one_img))

        pred_bbox = np.concatenate([
            np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
            np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
            np.reshape(pred_lbbox, (-1, 5 + self.num_classes))
        ], axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)
        print("Predicted boxes for {}: {}".format(image_name, bboxes))
        if self.isp_flag:
            print("ISP params:", isp_param)
            image_isped = utils.image_unpreporcess(image_isped[0, ...], [org_h, org_w])
            image_isped = np.clip(image_isped * 255, 0, 255)
        else:
            image_isped = np.clip(image, 0, 255)
        return bboxes, image_isped, time_one_img

    def evaluate(self):
        mAP_path = os.path.join(exp_folder, 'mAP')
        if not os.path.exists(mAP_path):
            os.makedirs(mAP_path)
        predicted_dir_path = os.path.join(mAP_path, 'predicted')
        ground_truth_dir_path = os.path.join(mAP_path, 'ground-truth')
        for d in [predicted_dir_path, ground_truth_dir_path, self.write_image_path]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.mkdir(d)
        time_total = 0
        num_img = 0
        with open(self.annotation_path, 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                annotation = line.strip().split()
                image_path = os.path.normpath(annotation[0])
                image_name = os.path.basename(image_path)
                image = cv2.imread(image_path)
                bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])
                if len(bbox_data_gt) == 0:
                    bboxes_gt, classes_gt = [], []
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                ground_truth_path = os.path.join(ground_truth_dir_path, f"{num}.txt")
                print("=> Ground truth of {}:".format(image_name))
                with open(ground_truth_path, 'w') as f:
                    for i in range(len(bboxes_gt)):
                        class_name = self.classes[classes_gt[i]]
                        bbox_mess = ' '.join([class_name] + list(map(str, bboxes_gt[i]))) + '\n'
                        f.write(bbox_mess)
                        print("\t" + bbox_mess.strip())
                print("=> Predict result of {}:".format(image_name))
                predict_result_path = os.path.join(predicted_dir_path, f"{num}.txt")
                t1 = time.time()
                bboxes_pr, image_isped, time_one_img = self.predict(image, image_name)
                num_img += 1
                time_total += time.time() - t1
                if self.write_image:
                    image_drawn = utils.draw_bbox(image_isped, bboxes_pr, self.classes, show_label=self.show_label)
                    cv2.imwrite(os.path.join(self.write_image_path, image_name), image_drawn)
                with open(predict_result_path, 'w') as f:
                    for bbox in bboxes_pr:
                        coor = np.array(bbox[:4], dtype=np.int32)
                        score = bbox[4]
                        class_ind = int(bbox[5])
                        class_name = self.classes[class_ind]
                        bbox_mess = ' '.join([class_name, f"{score:.4f}"] + list(map(str, coor))) + '\n'
                        f.write(bbox_mess)
                        print("\t" + bbox_mess.strip())
        print("**** Total processing time: {:.2f} sec for {} images, average time: {:.2f} sec/image".format(
            time_total, num_img, time_total / num_img))

if __name__ == '__main__':
    YoloTest().evaluate()
