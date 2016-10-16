# -*- coding: utf-8 -*-
import cv2
import numpy as np
import six

import datasets

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())

# Python 2 compatibility
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


def rect_center(rect):
    return np.array([rect[0] + rect[2] * 0.5,
                     rect[1] + rect[3] * 0.5])


def pt_in_rect(pt, rect):
    return (rect[0] <= pt[0] < rect[0] + rect[2] and
            rect[1] <= pt[1] < rect[1] + rect[3])


def select_closest_rect(pt, rects):
    closest = None
    min_sq_distance = float('inf')
    for rect in rects:
        diff = pt - rect_center(rect)
        sq_distance = diff[0] ** 2 + diff[1] ** 2
        if sq_distance < min_sq_distance:
            closest = rect
            min_sq_distance = sq_distance
    return closest


class CascadeFaceDetector(object):

    def __init__(self, cascade_paths):
        logger.info('Setup CascadeFaceDetector from {}'.format(cascade_paths))
        # face detector
        self.cascades = list()
        for path in cascade_paths:
            cascade = cv2.CascadeClassifier(path)
            if cascade.empty():
                logger.error('Failed to load cascade file (%s)', path)
            else:
                self.cascades.append(cascade)
        if len(self.cascades) == 0:
            logger.critical('Failed to setup, please check cascade paths.')

    def detect_joint_valid(self, img, joint, head_pt=None):
        if head_pt is None:
            head_pt = joint[datasets.JOINT_MAP['head']]
        # detect
        candidates = self._detect(img, largest_one=False)
        # find best rect
        candidates = list(filter(lambda x: pt_in_rect(head_pt, x), candidates))
        if len(candidates) == 0:
            # not found
            return None
        # select closest one
        facial_rect = select_closest_rect(head_pt, candidates)
        return facial_rect

    def detect_biggest(self, img):
        # detect
        candidates = self._detect(img, largest_one=True)
        # find biggest rect
        facial_rect = None
        max_size = 0
        for rect in candidates:
            size = rect[2] * rect[3]
            if size > max_size:
                facial_rect = rect
                max_size = size
        return facial_rect

    def _detect(self, img, largest_one=False):
        # color to gray
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # convert type
        if img.dtype == np.float32:
            img = (img * 255.0).astype(np.uint8)
        # histogram
        img = cv2.equalizeHist(img)
        # detect
        dst_rects = list()
        flags = cv2.CASCADE_SCALE_IMAGE
        if largest_one:
            flags |= cv2.CASCADE_FIND_BIGGEST_OBJECT
        for cascade in self.cascades:
            rects = cascade.detectMultiScale(img, scaleFactor=1.1,
                                             minNeighbors=3,
                                             minSize=(0, 0),
                                             maxSize=(100, 100),
                                             flags=flags)
            dst_rects.extend(rects)
        return np.array(dst_rects, copy=False)


class FaceBasedPoseNormalizer(object):

    def __init__(self):
        # normalization parameters
        self.param_shift = None
        self.param_scale = None

        # training parameters
        self.train_shift = list()
        self.train_scale = list()

    def calc_matrix(self, width, height, facial_rect, rect_scale=None):
        # check parameters
        if self.param_shift is None or self.param_scale is None:
            logger.error('No parameters')
            return None
        # Copy for safety
        facial_rect = np.array(facial_rect, copy=True, dtype=np.float32)

        # rect scaling
        if rect_scale is not None:
            w = facial_rect[2] * rect_scale
            h = facial_rect[3] * rect_scale
            facial_rect[0] -= (w - facial_rect[2]) * 0.5
            facial_rect[1] -= (h - facial_rect[3]) * 0.5
            facial_rect[2] = w
            facial_rect[3] = h

        # get joint rect
        shift = np.array(self.param_shift, copy=True)
        shift *= facial_rect[2:4]
        joint_center = rect_center(facial_rect) + shift
        scale = np.array(self.param_scale, copy=True)
        scale *= facial_rect[2:4]
        joint_root = joint_center - scale / 2

        # affine transform
        src_triangle = np.array([
            joint_root,
            [joint_root[0] + scale[0], joint_root[1]],
            [joint_root[0], joint_root[1] + scale[1]],
        ], dtype=np.float32)
        dst_triangle = np.array([
            [0, 0],
            [width, 0],
            [0, height],
        ], dtype=np.float32)
        affine_mat = cv2.getAffineTransform(src_triangle, dst_triangle)  # 2x3
        affine_mat = np.vstack((affine_mat, [0, 0, 1]))  # 3x3

        return affine_mat

    def clear(self):
        self.param_shift = None
        self.param_scale = None
        self.train_shift = list()
        self.train_scale = list()

    def save(self, filename):
        np.savez(filename, shift=self.param_shift, scale=self.param_scale)

    def load(self, filename):
        try:
            cache_data = np.load(filename)
            shift, scale = cache_data['shift'], cache_data['scale']
            self.param_shift, self.param_scale = shift, scale
            return True
        except (FileNotFoundError, KeyError):
            return False

    def train(self, train_dataset, max_size):
        ''' Train normalization parameters '''
        logger.info('Start to train pose normalizer')
        # Training loop
        train_size = min(len(train_dataset), max_size)
        for i in six.moves.xrange(train_size):
            # Show progress
            if i % 100 == 0:
                logger.info('  {} / {}'.format(i, train_size))
            # Load raw poses
            img, joint, facial_rect = train_dataset.get_raw(i)
            # Train normalizer
            self._train_one(img, joint, facial_rect)
        # Generate parameters
        self._gen_param()

    def _gen_param(self, scale_sigma=3.5):
        ''' Generate parameters from trained data
        `Scale sigma` is decided by [3-sigma + alpha]
        '''
        logger.info('Generate pose normalizer parameters')
        if len(self.train_shift) == 0 or len(self.train_scale) == 0:
            logger.error('No trained data')
            return
        # Convert to numpy array
        shift = np.asarray(self.train_shift)
        scale = np.asarray(self.train_scale)
        # Mean
        shift_mean = np.mean(shift, axis=0)
        scale_mean = np.mean(scale, axis=0)
        # Std
        scale_std = np.std(scale, axis=0)
        # Generate param
        self.param_shift = shift_mean
        self.param_scale = scale_mean + scale_sigma * scale_std
        # Clear
        self.train_shift = list()
        self.train_scale = list()

    def _train_one(self, img, joint, facial_rect):
        # Joint bounding rect
        joint_rect = cv2.boundingRect(joint)
        # face -> body shift
        shift = rect_center(joint_rect) - rect_center(facial_rect)
        shift /= np.asarray(facial_rect[2:4], dtype=np.float32)
        self.train_shift.append(shift)
        # face -> body scale
        scale = np.array([joint_rect[2], joint_rect[3]], dtype=np.float32)
        scale /= np.asarray(facial_rect[2:4], dtype=np.float32)
        self.train_scale.append(scale)
        # DEBUG: draw result rects
        # import drawing
        # drawing._draw_rect(img, facial_rect, (0, 0, 1))
        # drawing._draw_rect(img, joint_rect, (1, 0, 0))
        # cv2.imshow('img', img)
        # cv2.waitKey()
        return True


def setup_pose_normalizer(cache_path, train_dataset=None,
                          n_normalizer_sample=3000):
    ''' Setup FaceBasedPoseNormalizer
    When normalizer is trained, `train_dataset.get_raw(i)` will be called and
    should returns [img, joint, facial_rect].
    '''
    normalizer = FaceBasedPoseNormalizer()
    # Load
    logger.info('Try to load pose normalizer cache from {}'.format(cache_path))
    ret = normalizer.load(cache_path)
    if not ret:
        logger.info('Failed to load pose normalizer cache, so setup now')
        # Train
        if train_dataset is None:
            logger.critical('To train pose normalizer, `train_dataset` is'
                            ' needed')
            return None
        normalizer.train(train_dataset, n_normalizer_sample)
        # Save
        logger.info('Save pose normalizer to {}'.format(cache_path))
        normalizer.save(cache_path)
    return normalizer


def calc_y_flip_matrix(width, height):
    src_triangle = np.array([
        [0, 0],
        [width, 0],
        [0, height],
    ], dtype=np.float32)
    dst_triangle = np.array([
        [width, 0],
        [0, 0],
        [width, height],
    ], dtype=np.float32)
    affine_mat = cv2.getAffineTransform(src_triangle, dst_triangle)  # 2x3
    affine_mat = np.vstack((affine_mat, [0, 0, 1]))  # 3x3
    return affine_mat


def fix_back_shot(joint):
    jmap = datasets.JOINT_MAP
    if joint[jmap['lsho']][0] >= joint[jmap['rsho']][0]:
        return joint  # no flip

    dst_joint = np.empty_like(joint)
    for key, row in jmap.items():
        if key[0] == 'l':
            row2 = jmap['r' + key[1:]]
            dst_joint[row2] = joint[row]
        elif key[0] == 'r':
            row2 = jmap['l' + key[1:]]
            dst_joint[row2] = joint[row]
        else:
            dst_joint[row] = joint[row]
    return dst_joint


def transform_img(img, mat, width, height):
    img = cv2.warpAffine(img, mat[:2, :], (width, height))
    return img


def transform_joint(joint, mat):
    ''' Joint convert function
    Back shot joint will be fixed.
    '''
    joint = np.hstack((joint, np.ones((joint.shape[0], 1))))
    joint = mat.dot(joint.T).T
    joint = joint[:, 0:2]
    joint = fix_back_shot(joint)
    return joint


def transform_pose(img, joint, mat, width, height):
    ''' Pose convert function '''
    img = transform_img(img, mat, width, height)
    joint = transform_joint(joint, mat)
    return img, joint
