# -*- coding: utf-8 -*-
import chainer
import cv2
import numpy as np
import os
import os.path
import six

import datasets
# import drawing

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def mkdir_to_save(filename):
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def pt_in_rect(pt, rect):
    return (rect[0] <= pt[0] < rect[0] + rect[2] and
            rect[1] <= pt[1] < rect[1] + rect[3])


def rect_center(rect):
    return np.array([rect[0] + rect[2] / 2.0,
                     rect[1] + rect[3] / 2.0])


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


def boundingRect(points):
    min_pt = np.min(points, 0)
    max_pt = np.max(points, 0)
    diff = max_pt - min_pt
    return [min_pt[0], min_pt[1], diff[0], diff[1]]


def fix_back_shot(joint):
    jmap = datasets.JOINT_MAP
    if joint[jmap['lsho']][0] >= joint[jmap['rsho']][0]:
        return joint  # not flip

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


class FaceDetector(object):
    def __init__(self, cascade_paths):
        # face detector
        self.cascades = list()
        for path in cascade_paths:
            cascade = cv2.CascadeClassifier(path)
            if cascade.empty():
                logger.error('Failed to load cascade file (%s)', path)
            else:
                self.cascades.append(cascade)

    def detect_joint_valid_face(self, img, joint, head_pt=None):
        if head_pt is None:
            head_pt = joint[datasets.JOINT_MAP['head']]
        # detect
        candidates = self._detect_face(img, largest_one=False)
        # find best rect
        candidates = list(filter(lambda x: pt_in_rect(head_pt, x), candidates))
        if len(candidates) == 0:
            # not found
            return None
        # select closest one
        facial_rect = select_closest_rect(head_pt, candidates)
        return facial_rect

    def detect_biggest_face(self, img):
        # detect
        candidates = self._detect_face(img, largest_one=True)
        # find biggest rect
        facial_rect = None
        max_size = 0
        for rect in candidates:
            size = rect[2] * rect[3]
            if size > max_size:
                facial_rect = rect
                max_size = size
        return facial_rect

    def _detect_face(self, img, largest_one=False):
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
        logger.info('Save pose normalizer parameters (%s)', filename)
        mkdir_to_save(filename)
        np.save(filename, [self.param_shift, self.param_scale])

    def load(self, filename):
        logger.info('Load pose normalizer parameters (%s)', filename)
        try:
            self.param_shift, self.param_scale = np.load(filename)
            return True
        except:
            logger.error('Failed to load pose normalizer parameters')
            return False

    def train(self, data_loader, face_detector, max_size=1000):
        ''' Train normalization parameters '''
        logger.info('Train pose normalizers')
        # train loop
        train_size = min(data_loader.get_size(), max_size)
        failed_cnt = 0
        for i in six.moves.xrange(train_size):
            # show progress
            if i % 10 == 0:
                logger.info('  %d / %d, (failed: %d)',
                            i, train_size, failed_cnt)
            # load poses
            img, joint = data_loader.get_data(i)
            # train normalizer
            ret = self._train_one(img, joint, face_detector)
            if not ret:
                failed_cnt += 1
        # generate parameters
        self._generate_param()

    def _generate_param(self, scale_sigma=3.0):
        ''' Generate parameters from trained data '''
        logger.info('Generate FaceBasedPoseNormalizer parameters')
        if len(self.train_shift) == 0 or len(self.train_scale) == 0:
            logger.error('No trained data')
            return
        # convert to numpy array
        shift = np.array(self.train_shift, copy=False)
        scale = np.array(self.train_scale, copy=False)
        # mean
        shift_mean = np.mean(shift, axis=0)
        scale_mean = np.mean(scale, axis=0)
        # std
        scale_std = np.std(scale, axis=0)
        # generate param
        self.param_shift = shift_mean
        self.param_scale = scale_mean + scale_sigma * scale_std
        # clear
        self.train_shift = list()
        self.train_scale = list()

    def _train_one(self, img, joint, face_detector):
        # detect face
        head_pt = joint[datasets.JOINT_MAP['head']]
        facial_rect = face_detector.detect_joint_valid_face(img, joint, head_pt)
        if facial_rect is None:
            return False
        # joint bounding rect
        joint_rect = boundingRect(joint)
        # face -> body shift
        shift = rect_center(joint_rect) - rect_center(facial_rect)
        shift /= facial_rect[2:4]
        self.train_shift.append(shift)
        # face -> body scale
        scale = np.array([joint_rect[2], joint_rect[3]])
        scale /= facial_rect[2:4]
        self.train_scale.append(scale)
        # DEBUG: draw result rects
        # drawing.draw_rect(img, facial_rect, (0, 0, 1))
        # drawing.draw_rect(img, joint_rect, (1, 0, 0))
        # cv2.imshow('img', img)
        # cv2.waitKey()
        return True


def calc_cropping_matrix(width, height, center, joint, sigma=1.0):
    jmap = datasets.JOINT_MAP
    if 'rhip' in jmap and 'lhip' in jmap:
        diam1 = np.sqrt(np.sum((joint[jmap['lsho']] - joint[jmap['rhip']]) ** 2))
        diam2 = np.sqrt(np.sum((joint[jmap['rsho']] - joint[jmap['lhip']]) ** 2))
        diam = (diam1 + diam2) / 2.0
    else:
        diam = np.sqrt(np.sum((joint[jmap['lsho']] - joint[jmap['rsho']]) ** 2))
        diam *= 2.0
    diam *= sigma
    half_diam = diam / 2.0

    # affine transform
    src_triangle = np.array([
        [center[0] - half_diam, center[1] - half_diam],
        [center[0] + half_diam, center[1] - half_diam],
        [center[0] - half_diam, center[1] + half_diam],
    ], dtype=np.float32)
    dst_triangle = np.array([
        [0, 0],
        [width, 0],
        [0, height],
    ], dtype=np.float32)
    affine_mat = cv2.getAffineTransform(src_triangle, dst_triangle)  # 2x3
    affine_mat = np.vstack((affine_mat, [0, 0, 1]))  # 3x3

    return affine_mat


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


def transform_joint_pt(joint_pt, mat, translate=True):
    ''' Joint diff point convert function '''
    if translate:
        joint_pt = np.hstack((joint_pt, 1))
    else:
        joint_pt = np.hstack((joint_pt, 0))
    joint_pt = mat.dot(joint_pt.T).T
    joint_pt = joint_pt[0:2]
    return joint_pt


def transform_pose(img, joint, mat, width, height):
    ''' Pose convert function '''
    img = transform_img(img, mat, width, height)
    joint = transform_joint(joint, mat)
    return img, joint


def calc_flip_matrix(width, height):
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


def conv_imgs_to_chainer(xp, raw_imgs, train=False):
    ''' Convert numpy images to chainer variable '''
    # Send to GPU
    xp_imgs = xp.asarray(raw_imgs, dtype=np.float32)
    # image: (n, w, h, c) -> (n, c, w, h)
    xp_imgs = xp.transpose(xp_imgs, (0, 3, 1, 2))
    # Chainer variable
    x = chainer.Variable(xp_imgs, volatile=not train)
    return x


def conv_joints_to_chainer(xp, raw_joints, xp_img_shape, train=False,
                           scale_mode=''):
    ''' Convert numpy joints to chainer variable '''
    # Send to GPU
    xp_joints = xp.asarray(raw_joints, dtype=np.float32)
    # to [-0.5:0:5]
    if scale_mode == '':
        pass
    elif scale_mode == '+':
        xp_joints = (xp_joints - xp_img_shape / 2.0) / xp_img_shape
    elif scale_mode == '+-':
        xp_joints = xp_joints / xp_img_shape
    else:
        logger.error('Not implemented mode: %s', str(scale_mode))
    # (n, j, 2) -> (n, j * 2)
    xp_joints = xp_joints.reshape(xp_joints.shape[0], -1)
    # Chainer variable
    x = chainer.Variable(xp_joints, volatile=not train)
    return x


def conv_joints_from_chainer(xp, pred, xp_img_shape, scale_mode=''):
    ''' Convert numpy joints to chainer variable '''
    xp_joints = pred.data
    xp_joints = xp_joints.reshape((xp_joints.shape[0], -1, 2))
    # from [-0.5:0:5]
    if scale_mode == '':
        pass
    elif scale_mode == '+':
        xp_joints = (xp_joints * xp_img_shape) + xp_img_shape / 2.0
    elif scale_mode == '+-':
        xp_joints = xp_joints * xp_img_shape
    else:
        logger.error('Not implemented mode: %s', str(scale_mode))

    if hasattr(xp_joints, 'get'):
        pred_joints = xp_joints.get()  # to cpu
    else:
        pred_joints = xp_joints
    return pred_joints
