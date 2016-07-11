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
import normalizers
import model_io
from preparation import precompute_params
import settings

# logging
from logging import getLogger, INFO, DEBUG
log_initializer.setFmt()
log_initializer.setRootLevel(DEBUG)
logger = getLogger(__name__)
imgviewer.logger.setLevel(INFO)


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--resume', dest='resume', action='store_true')
parser.add_argument('--stage', dest='stage', type=int, default=0)
parser.add_argument('--joint_idx', dest='joint_idx', type=int, default=-1)
args = parser.parse_args()
assert(args.stage >= 0)

GPU = settings.GPU
RESUME = args.resume
STAGE = args.stage
JOINT_IDX = args.joint_idx

if STAGE != 0 and not 0 <= JOINT_IDX < len(datasets.JOINT_MAP):
    logger.critical('Invalid joint_idx (%d)', JOINT_IDX)
    exit()

# Setup GPU
if GPU >= 0:
    cuda.check_cuda_available()
    logger.info('GPU mode (%d) (stage: %d, joint_idx: %d, resume: %r)',
                GPU, STAGE, JOINT_IDX, RESUME)
else:
    logger.info('CPU mode (stage: %d, joint_idx: %d, resume: %r)',
                STAGE, JOINT_IDX, RESUME)
xp = cuda.cupy if GPU >= 0 else np


def create_precomp_modifier(stage_cnt, joint_idx):
    if stage_cnt == 0 or stage_cnt == 1:
        return "_s%d" % stage_cnt
    else:
        return "_s%d_j%d" % (stage_cnt, joint_idx)


if __name__ == '__main__':
    # Previous loaders
    if STAGE == 0:
        # Load flic dataset
        flic = datasets.Flic()
        flic.load(settings.FLIC_FULL_PATH, settings.FLIC_PLUS_PATH)
        # DEBUG: limit data size
        # flic.train_data.limit_size(20)
        # flic.test_data.limit_size(20)
        # Set as previous loader
        prev_train_loader = flic.train_data
        prev_test_loader = flic.test_data
    else:
        # Previous stage loader
        prev_train_loader = datasets.PoseDataLoader()
        prev_test_loader = datasets.PoseDataLoader()
        modif = create_precomp_modifier(STAGE - 1, JOINT_IDX)
        ret1 = prev_train_loader.load(settings.PRECOMP_TRAIN % modif)
        ret2 = prev_test_loader.load(settings.PRECOMP_TEST % modif)
        if not ret1 or not ret2:
            logger.critical('Previous stage parameters are not found.')
            exit()

    if STAGE == 0 or STAGE == 1:
        # Face Detector
        detector = normalizers.FaceDetector(settings.CASCADE_PATHS)
        # Pose Normalizer
        normalizer = convenient.get_inited_pose_normalizer(prev_train_loader,
                                                           detector)
    else:
        detector = None
        normalizer = None

    # Setup train loader
    train_loader = datasets.PoseDataLoader()
    modif = create_precomp_modifier(STAGE, JOINT_IDX)
    ret = train_loader.load(settings.PRECOMP_TRAIN % modif)
    if not ret:
        # Precompute predicted joints
        loader = precompute_params(xp, STAGE, JOINT_IDX, prev_train_loader,
                                   normalizer, detector)
        # Save
        loader.save(settings.PRECOMP_TRAIN % modif)
        # Set
        train_loader = loader

    # Setup test loader
    test_loader = datasets.PoseDataLoader()
    modif = create_precomp_modifier(STAGE, JOINT_IDX)
    ret = test_loader.load(settings.PRECOMP_TEST % modif)
    if not ret:
        # Precompute predicted joints
        loader = precompute_params(xp, STAGE, JOINT_IDX, prev_test_loader,
                                   normalizer, detector)
        # Save
        loader.save(settings.PRECOMP_TEST % modif)
        # Set
        test_loader = loader

    # Train data load event and queue
    train_data_que = multiprocessing.Queue(settings.N_BUFFERING_BATCH)
    train_load_evt = multiprocessing.Event()
    # Test data load event and queue
    test_data_que = multiprocessing.Queue(settings.N_BUFFERING_BATCH)
    test_load_evt = multiprocessing.Event()
    # Image server queue
    server_que = multiprocessing.Queue()
    # Visualizer data queue
    visual_que = multiprocessing.Queue()
    pose_comp_que = multiprocessing.Queue()  # for stage >= 1

    # First pose convert function
    def first_conv_func(img, joint, param, tag):
        facial_rect = param
        width, height = alexnet.IMG_WIDTH, alexnet.IMG_HEIGHT
        rect_scale = None
        # Random cropping
        if tag == 'train':
            rect_scale = np.random.normal(loc=1.0, scale=0.3)
            rect_scale = min(max(rect_scale, 0.95), 1.1)  # clamp
        mat = normalizer.calc_matrix(width, height, facial_rect,
                                     rect_scale=rect_scale)
        # Random flipping
        if tag == 'train' and np.random.randint(2) == 0:
            mat = normalizers.calc_flip_matrix(width, height).dot(mat)
        # Apply
        img, joint = normalizers.transform_pose(img, joint, mat, width, height)
        return img, joint

    if STAGE > 0:
        logger.info('Sum up joint differences')
        diffs = list()

        # Sum up the difference
        def conv_func(img, joint, param):
            pred_joint = param
            # Fix back shots
            joint = normalizers.fix_back_shot(joint)
            pred_joint = normalizers.fix_back_shot(pred_joint)
            # Predicting difference
            diff = joint[JOINT_IDX] - pred_joint[JOINT_IDX]
            diffs.append(diff)

        # Apply for all test poses
        for i in six.moves.xrange(test_loader.get_size()):
            test_loader.get_data(i, conv_func)
        # Calculate 2D parameters
        diffs = np.asarray(diffs)
        diff_mean = np.mean(diffs, axis=0)  # [x, y]
        diff_std = np.sqrt(np.mean((diffs - diff_mean) ** 2, axis=0))
        logger.info('Difference mean: %ss', str(diff_mean))
        logger.info('Difference std: %s', str(diff_std))

    # Subsequent pose convert function
    def subseq_conv_func(img, joint, param, tag):
        pred_joint = param
        width, height = alexnet.IMG_WIDTH, alexnet.IMG_HEIGHT
        # Fix back shot
        joint = normalizers.fix_back_shot(joint)
        pred_joint = normalizers.fix_back_shot(pred_joint)
        center = pred_joint[JOINT_IDX]
        # Padding
        if tag == 'train':
            # Add noise
            center[0] += np.random.normal(loc=diff_mean[0], scale=diff_std[0])
            center[1] += np.random.normal(loc=diff_mean[1], scale=diff_std[1])
        mat = normalizers.calc_cropping_matrix(width, height, center,
                                               pred_joint,
                                               sigma=settings.BBOX_SIGMA)
        # Apply
        new_img = normalizers.transform_img(img, mat, width, height)
        new_center = normalizers.transform_joint_pt(center, mat)
        new_joint_pt = normalizers.transform_joint_pt(joint[JOINT_IDX], mat)
        # Send the difference as teacher
        diff = new_joint_pt - new_center
        diff = diff.reshape((1,) + diff.shape)  # [x, y] -> [[x, y]]
        return new_img, diff

    # States loader
    def load_states_func():
        if RESUME:
            return model_io.load_states(STAGE, JOINT_IDX)
        else:
            return model_io.setup_initial_states(STAGE)

    # States saver
    def save_states_func(epoch_cnt, model, optimizer, train_losses,
                         test_losses):
        model_io.save_states(STAGE, JOINT_IDX, epoch_cnt, model, optimizer,
                             train_losses, test_losses)

    if STAGE == 0:
        conv_func = first_conv_func
        joint_scale_mode = '+'
    else:
        conv_func = subseq_conv_func
        joint_scale_mode = '+-'

    # Train data loader process
    start_process(loops.load_pose_loop, 'train', train_data_que, train_load_evt,
                  train_loader, settings.BATCH_SIZE, conv_func, True)
    # Test data loader process
    start_process(loops.load_pose_loop, 'test', test_data_que, test_load_evt,
                  test_loader, settings.BATCH_SIZE, conv_func, True)
    # Training loop process
    start_process(loops.train_pose_loop, xp, settings.N_EPOCH,
                  train_data_que, train_load_evt,
                  test_data_que, test_load_evt, visual_que,
                  load_states_func, save_states_func, joint_scale_mode)
    # Image server process
    imgviewer.start(server_que, stop_page=False, port=settings.SERVER_PORT)
    # Visualizer loop process
    start_process(loops.visualize_pose_loop, visual_que, server_que,
                  settings.N_VIEW_IMG)

    # Wait for exit
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt, exit...')
