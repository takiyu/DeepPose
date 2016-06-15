#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import multiprocessing
import numpy as np
import six
import time

from chainer import cuda

import alexnet
import convenient
from convenient import start_process
import datasets
from image_servers import imgviewer
import log_initializer
import loops
import model_io
import normalizers
import settings

# logging
from logging import getLogger, DEBUG, INFO
log_initializer.setFmt()
log_initializer.setRootLevel(DEBUG)
logger = getLogger(__name__)
imgviewer.logger.setLevel(INFO)


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--stage', dest='stage', type=int, default=0, metavar='N')
args = parser.parse_args()
assert(args.stage >= 0)

N_SHOW = 6
stage = args.stage
GPU = settings.GPU
SLEEP_SEC = 0.0

# Setup GPU
if GPU >= 0:
    cuda.check_cuda_available()
    logger.info('GPU mode (%d) (stage: %d)', GPU, stage)
else:
    logger.info('CPU mode (stage: %d)', stage)
xp = cuda.cupy if GPU >= 0 else np


def use_model_single(stage_cnt, joint_idx, model, img, detector=None,
                     normalizer=None, pre_joint=None, teacher=None):
    width, height = alexnet.IMG_WIDTH, alexnet.IMG_HEIGHT
    xp_img_shape = xp.asarray([width, height], dtype=np.float32)

    # Normalize
    if stage_cnt == 0:
        facial_rect = detector.detect_joint_valid_face(img, teacher)
        # facial_rect = detector.detect_biggest_face(img)
        if facial_rect is None:
            logger.info('Failed to detect face')
            return None
        mat = normalizer.calc_matrix(width, height, facial_rect)
    else:
        center = pre_joint[joint_idx]
        mat = normalizers.calc_cropping_matrix(width, height, center, pre_joint,
                                               sigma=settings.BBOX_SIGMA)
    img = normalizers.transform_img(img, mat, width, height)

    # img -> imgs
    imgs = img.reshape((1,) + img.shape)
    # numpy -> chainer variable
    x = normalizers.conv_imgs_to_chainer(xp, imgs, train=False)
    # Use model
    pred = model(x)
    # chainer variable -> numpy
    if stage_cnt == 0:
        joint_scale_mode = '+'
    else:
        joint_scale_mode = '+-'
    pred_joints = normalizers.conv_joints_from_chainer(xp, pred,
                                                       xp_img_shape,
                                                       joint_scale_mode)

    # joints -> joint
    pred_joint = pred_joints[0]
    inv_mat = np.linalg.inv(mat)

    # Denormalize
    if stage_cnt == 0:
        pred_joint = normalizers.transform_joint(pred_joint, inv_mat)
        return pred_joint
    else:
        assert(len(pred_joint) == 1)
        diff_pt = pred_joint[0]
        diff_pt = normalizers.transform_joint_pt(diff_pt, inv_mat,
                                                 translate=False)
        return diff_pt


if __name__ == '__main__':
    logger.info('Use trained models')

    # Load flic dataset
    flic = datasets.Flic()
    flic.load(settings.FLIC_FULL_PATH, settings.FLIC_PLUS_PATH)
    loader = flic.test_data

    # Face Detector
    detector = normalizers.FaceDetector(settings.CASCADE_PATHS)
    # Pose Normalizer
    normalizer = convenient.get_inited_pose_normalizer(flic.train_data,
                                                       detector)

    # Image server process
    server_que = multiprocessing.Queue()
    imgviewer.start(server_que, stop_page=False, port=settings.SERVER_PORT)
    # Visualizer loop process
    visual_que = multiprocessing.Queue()
    start_process(loops.visualize_pose_loop, visual_que, server_que)

    # Draw randomly
    show_cnt = 0
    perm = np.random.permutation(flic.test_data.get_size())
    for i in six.moves.xrange(perm.shape[0]):
        # Load pose
        logger.info('Load FLIC test %dth image', perm[i])
        raw_img, teacher = flic.test_data.get_data(perm[i])
        # Teacher data is used only for face detection check

        # Load first model and use
        first_model = model_io.load_best_model(0, -1)
        pred_joint = use_model_single(0, -1, first_model, raw_img,
                                      detector=detector, normalizer=normalizer,
                                      teacher=teacher)
        if pred_joint is None:
            continue

        results = [pred_joint]

        # Subsequent models
        for joint_idx in six.moves.xrange(len(datasets.JOINT_MAP)):
            stage_cnt = 0
            while True:
                stage_cnt += 1
                # Try to load next stage model
                try:
                    model = model_io.load_best_model(stage_cnt, joint_idx)
                except FileNotFoundError:
                    logger.info('Failed to load')
                    break
                # Use model
                diff_pt = use_model_single(stage_cnt, joint_idx, model, raw_img,
                                           pre_joint=results[stage_cnt - 1])
                # Create new result
                if len(results) <= stage_cnt:
                    next_joint = np.array(results[stage_cnt - 1], copy=True)
                    results.append(next_joint)
                # Apply
                results[stage_cnt][joint_idx] += diff_pt

        # Show
        logger.info('Show FLIC test %dth image', perm[i])
        for stage_cnt, result in enumerate(results):
            # single -> multi
            imgs = raw_img.reshape((1,) + raw_img.shape)
            joints = result.reshape((1,) + result.shape)
            # Send to server
            tab_name = "test (stage %d)" % stage_cnt
            data = [show_cnt % N_SHOW, imgs, joints]
            visual_que.put(('pose', tab_name, data))

        show_cnt += 1
        if SLEEP_SEC > 0.0:
            time.sleep(SLEEP_SEC)
