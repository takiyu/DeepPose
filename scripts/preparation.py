# -*- coding: utf-8 -*-

from convenient import start_process
import six
import multiprocessing
import numpy as np

import alexnet
import datasets
import loops
import model_io
import normalizers
import settings

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())

GPU = settings.GPU
STAGE = None
JOINT_IDX = None

# Shared variables
new_params = None
mat_que = None
base_joint_que = None
face_normalizer = None
out_que = None


# Load model func
def load_model_func():
    return model_io.load_best_model(STAGE - 1, JOINT_IDX, gpu=GPU >= 0,
                                    train=False)


# First pose convert function
def first_conv_func(img, joint, param, tag):
    facial_rect = param
    width, height = alexnet.IMG_WIDTH, alexnet.IMG_HEIGHT
    mat = face_normalizer.calc_matrix(width, height, facial_rect)
    # Send matrix
    mat_que.put(mat)
    # Apply
    img, dummy_joint = normalizers.transform_pose(img, joint, mat,
                                                  width, height)
    return img, dummy_joint


# First pose register function
def first_register_loop():
    while True:
        data = out_que.get()
        if data is None:
            break
        pred_joints = data
        # First pose register
        for joint in pred_joints:
            mat = mat_que.get()
            inv_mat = np.linalg.inv(mat)
            joint = normalizers.transform_joint(joint, inv_mat)
            # Register
            new_params.append(joint)


# Subsequent pose convert function
def subseq_conv_func(img, joint, param, tag):
    pred_joint = param
    width, height = alexnet.IMG_WIDTH, alexnet.IMG_HEIGHT
    # Crop pose at current joint index
    center = pred_joint[JOINT_IDX]
    mat = normalizers.calc_cropping_matrix(width, height, center,
                                           pred_joint,
                                           sigma=settings.BBOX_SIGMA)
    # Send matrix
    mat_que.put(mat)
    # Apply
    new_img = normalizers.transform_img(img, mat, width, height)
    # Send base joint
    base_joint_que.put(pred_joint)
    # Send the difference as teacher (dummy)
    dummy_diff = np.asarray([0, 0])
    return new_img, dummy_diff


# Subsequent pose convert function
def subseq_register_loop():
    while True:
        data = out_que.get()
        if data is None:
            break
        diff_pts = data
        # Subsequent pose register function
        for diff_pt in diff_pts:
            assert(len(diff_pt) == 1)
            diff_pt = diff_pt[0]
            mat = mat_que.get()
            inv_mat = np.linalg.inv(mat)
            diff_pt = normalizers.transform_joint_pt(diff_pt, inv_mat)
            # Receive base joint
            base_joint = base_joint_que.get()
            # Merge
            base_joint[JOINT_IDX] += diff_pt
            # Register
            new_params.append(base_joint)


def precompute_params(xp, stage, joint_idx, prev_loader, normalizer=None,
                      detector=None):
    # Set variables to global
    global STAGE, JOINT_IDX
    global new_params, mat_que, base_joint_que, face_normalizer, out_que
    STAGE = stage
    JOINT_IDX = joint_idx
    face_normalizer = normalizer

    # First stage
    if stage == 0:
        new_params = list()
        valid_idx = list()
        logger.info('Precompute facial rects')
        failed_cnt = 0
        for i in six.moves.xrange(prev_loader.get_size()):
            # show progress
            if i % 10 == 0:
                logger.info('  %d / %d (failed: %d)',
                            i, prev_loader.get_size(), failed_cnt)
            # load image
            img, joint = prev_loader.get_data(i)
            # get param
            facial_rect = detector.detect_joint_valid_face(img, joint)
            if facial_rect is None:
                failed_cnt += 1
                continue
            # append
            new_params.append(facial_rect)
            valid_idx.append(i)

        # Create new loader
        loader = datasets.PoseDataLoader()
        loader.set_from_raw(np.asarray(prev_loader.img_paths)[valid_idx],
                            np.asarray(prev_loader.joints)[valid_idx],
                            new_params)
        return loader

    # Subsequent stage
    else:
        logger.info('Precompute predicted joints')
        new_params = list()

        # Train data load event and queue
        data_que = multiprocessing.Queue(settings.N_BUFFERING_BATCH)
        load_evt = multiprocessing.Event()
        # Predicted joint queue
        out_que = multiprocessing.Queue()
        # Matrix queue
        mat_que = multiprocessing.Queue()
        base_joint_que = multiprocessing.Queue()  # for stage >= 2

        # Stage selection
        if stage - 1 == 0:
            conv_func = first_conv_func
            register_loop = first_register_loop
            joint_scale_mode = '+'
        else:
            conv_func = subseq_conv_func
            register_loop = subseq_register_loop
            joint_scale_mode = '+-'

        # Data loader thread
        start_process(loops.load_pose_loop, 'predict', data_que, load_evt,
                      prev_loader, settings.BATCH_SIZE, conv_func, False)
        # Prediction loop process
        start_process(loops.predict_pose_loop, xp, data_que, load_evt,
                      out_que, load_model_func, joint_scale_mode)

        # Receive predicted joints and register params
        register_loop()

        # Register predicted joints and matrix as parameters
        loader = datasets.PoseDataLoader()
        loader.set_from_raw(prev_loader.img_paths, prev_loader.joints,
                            new_params)
        return loader
