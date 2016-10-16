# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os.path
import scipy.io

import chainer

import common

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())

# Constant variables
N_LANDMARK = 21
IMG_SIZE = (227, 227)

# Python 2 compatibility
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


def _rect_contain(rect, pt):
    x, y, w, h = rect
    return x <= pt[0] <= x + w and y <= pt[1] <= y + h

# Common joint indices
JOINT_MAP = {
    'lsho': 0,  # L_Shoulder
    'lelb': 1,  # L_Elbow
    'lwri': 2,  # L_Wrist
    'rsho': 3,  # R_Shoulder
    'relb': 4,  # R_Elbow
    'rwri': 5,  # R_Wrist
    'lhip': 6,  # L_Hip
    'rhip': 7,  # R_Hip
    'head': 8,  # Head
}

# Raw joint indices in FLIC dataset
FLIC_RAW_JOINT_MAP = {
    'lsho': 0,  # L_Shoulder
    'lelb': 1,  # L_Elbow
    'lwri': 2,  # L_Wrist
    'rsho': 3,  # R_Shoulder
    'relb': 4,  # R_Elbow
    'rwri': 5,  # R_Wrist
    'lhip': 6,  # L_Hip
    'rhip': 9,  # R_Hip
    'leye': 12,  # L_Eye
    'reye': 13,  # R_Eye
    'nose': 16,  # Nose
}

def _fix_raw_flic_joint(raw_coords):
    ''' Extract joint coordinates based on JOINT_MAP '''
    joint = np.empty((len(JOINT_MAP), 2), dtype=np.float32)
    for key, row in JOINT_MAP.items():
        if key == 'head':
            # head coordinate
            raw_idx0 = FLIC_RAW_JOINT_MAP['leye']
            raw_idx1 = FLIC_RAW_JOINT_MAP['reye']
            raw_idx2 = FLIC_RAW_JOINT_MAP['nose']
            joint[row] = (raw_coords[raw_idx0] +
                          raw_coords[raw_idx1] +
                          raw_coords[raw_idx2]) / 3
        else:
            raw_idx = FLIC_RAW_JOINT_MAP[key]
            joint[row] = raw_coords[raw_idx]
    return joint


def _load_raw_flic(flic_full_path, tr_plus_indices_path, train):
    ''' Load raw FLIC dataset from sqlite file
    Return:
        [dict('pose_id', 'img_path', 'joint')]
    '''
    logger.info('Load raw FLIC dataset from {}'.format(flic_full_path))
    dataset = list()

    # Load example mat
    examples_path = os.path.join(flic_full_path, "examples.mat")
    examples = scipy.io.loadmat(examples_path)
    examples = examples['examples'].reshape(-1)

    # Load training data indices
    train_indices = scipy.io.loadmat(tr_plus_indices_path)
    train_indices = train_indices['tr_plus_indices']

    # Image directory
    img_dir = os.path.join(flic_full_path, "images")

    # Register each data
    for i, example in enumerate(examples):
        # Check train or test
        if (i in train_indices) is train:
            # filename
            filename = example[3][0]
            img_path = os.path.join(img_dir, filename)
            # joint
            coordinates = example[2].T
            joint = _fix_raw_flic_joint(coordinates)
            # Register
            data = {'pose_id': i, 'img_path': img_path, 'joint': joint}
            dataset.append(data)

    return dataset


class FLIC(chainer.dataset.DatasetMixin):
    ''' FLIC Dataset
    '''

    def __init__(self, face_detector):
        chainer.dataset.DatasetMixin.__init__(self)

        # Member variables
        self.dataset = list()
        self.face_detector = face_detector
        self.pose_normalizer = None

    def setup_raw(self, flic_full_path, flic_plus_path, train):
        dst_dataset = list()

        # Load raw dataset
        raw_dataset = _load_raw_flic(flic_full_path, flic_plus_path, train)

        # Precompute facial rectangles (remove invalid entries)
        logger.info('Precompute facial rectangles')
        valid_cnt = 0
        for i, entry in enumerate(raw_dataset):
            # Logging
            if i % 10 == 0:
                logger.info(' {}/{} (valid: {})'
                            .format(i, len(raw_dataset), valid_cnt))
            # Entry variables
            joint = entry['joint']
            img = cv2.imread(entry['img_path'])
            if img is None or img.size == 0:
                continue  # skip
            # Detect a valid facial rectangle
            head_pt = joint[JOINT_MAP['head']]
            facial_rect = self.face_detector.detect_joint_valid(img, joint,
                                                                head_pt)
            if facial_rect is not None:
                entry['facial_rect'] = facial_rect
                dst_dataset.append(entry)
                valid_cnt += 1

        # Register
        self.dataset = dst_dataset

    def __len__(self):
        return len(self.dataset)

    def set_normalizer(self, pose_normalizer):
        self.pose_normalizer = pose_normalizer

    def get_raw(self, i):
        entry = self.dataset[i]
        img_path = entry['img_path']
        joint = entry['joint']
        facial_rect = entry['facial_rect']
        # Load image
        img = cv2.imread(img_path)
        if img is None or img.size == 0:
            logger.warn('Invalid image "{}"'.format(img_path))
            raise IndexError
        return img, joint, facial_rect

    def get_example(self, i):
        # Load raw data
        img, joint, facial_rect = self.get_raw(i)

        img_w, img_h = IMG_SIZE

        # === Pose Normalize (resize to IMG_SIZE) ===
        if self.pose_normalizer is not None:
            # Random cropping
            rect_scale = np.random.normal(loc=1.00, scale=0.3)
            rect_scale = min(max(rect_scale, 0.95), 1.05)  # clamp
            mat = self.pose_normalizer.calc_matrix(img_w, img_h, facial_rect,
                                                   rect_scale=rect_scale)
            # Random flipping
            if np.random.randint(2) == 0:
                mat = common.calc_y_flip_matrix(img_w, img_h).dot(mat)
        # Apply
        img, joint = common.transform_pose(img, joint, mat, img_w, img_h)

        # === Convert for Chainer types ===
        # Image [0:255](about) -> [-0.5:0.5]
        img = img.astype(np.float32)
        img = cv2.normalize(img, None, -0.5, 0.5, cv2.NORM_MINMAX)
        img = np.transpose(img, (2, 0, 1))
        # Joint  ([0:img_w,img_h] -> [-0.5:0.5])
        joint_offset = np.array([img_w / 2, img_h / 2], dtype=np.float32)
        joint_denom = np.array([img_w, img_h], dtype=np.float32)
        joint = (joint - joint_offset) / joint_denom
        # [n_joint, 2] -> [n_joint * 2]
        joint = joint.reshape(-1)
        joint = joint.astype(np.float32)

        return {'x_img': img, 't_joint': joint}


def setup_flic(cache_path, face_detector, flic_full_path=None,
               flic_plus_path=None):
    # Empty FLIC
    flic_train = FLIC(face_detector)
    flic_test = FLIC(face_detector)

    logger.info('Try to load FLIC cache from "{}"'.format(cache_path))
    try:
        # Load cache
        cache_data = np.load(cache_path)
        dataset_train = cache_data['dataset_train'].tolist()
        dataset_test = cache_data['dataset_test'].tolist()
        # Set to FILC
        flic_train.dataset = dataset_train
        flic_test.dataset = dataset_test
        logger.info('Succeeded in loading FLIC cache')

    except (FileNotFoundError, KeyError):
        # Setup FLIC
        logger.info('Failed to load FLIC cache, so setup now')
        if flic_full_path is None or flic_plus_path is None:
            logger.critical('`flic_full_path` and `flic_plus_path` are needed'
                            ' to load raw FLIC')
        flic_train.setup_raw(flic_full_path, flic_plus_path, train=True)
        logger.info('FLIC dataset (train): {}'.format(len(flic_train)))
        flic_test.setup_raw(flic_full_path, flic_plus_path, train=False)
        logger.info('FLIC dataset (test): {}'.format(len(flic_test)))

        # Save cache
        logger.info('Save FLIC cache to "{}"'.format(cache_path))
        np.savez(cache_path, dataset_train=flic_train.dataset,
                 dataset_test=flic_test.dataset)

    # Split dataset into train and test
    logger.info('FLIC datasets (n_train:{}, n_test:{})'.
                format(len(flic_train), len(flic_test)))

    return flic_train, flic_test
