# -*- coding: utf-8 -*-
import numpy as np
import os.path
import scipy.io

from .loader import PoseDataLoader

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


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


class Flic(object):
    ''' FLIC Dataset '''
    # using joint indices
    raw_joint_map = {
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

    def __init__(self):
        self.train_data = PoseDataLoader()
        self.test_data = PoseDataLoader()

    def load(self, flic_full_path, tr_plus_indices_path):
        logger.info('Load FLIC dataset (%s)', flic_full_path)

        # load example mat
        examples_path = os.path.join(flic_full_path, "examples.mat")
        examples = scipy.io.loadmat(examples_path)
        examples = examples['examples'].reshape(-1)

        # load training data indices
        train_indices = scipy.io.loadmat(tr_plus_indices_path)
        train_indices = train_indices['tr_plus_indices']

        # image directory
        img_dir = os.path.join(flic_full_path, "images")

        # data lists
        train_img_path_list = list()
        train_joint_list = list()
        test_img_path_list = list()
        test_joint_list = list()

        # append to each data list
        for i, example in enumerate(examples):
            # filename
            filename = example[3][0]
            img_path = os.path.join(img_dir, filename)

            # joint
            coordinates = example[2].T
            joint = self._extract_joint(coordinates)

            # register
            if i in train_indices:
                train_img_path_list.append(img_path)
                train_joint_list.append(joint)
            else:
                test_img_path_list.append(img_path)
                test_joint_list.append(joint)

        # setup PoseDataLoader
        self.train_data.set_from_raw(train_img_path_list, train_joint_list)
        self.test_data.set_from_raw(test_img_path_list, test_joint_list)

        logger.info('Finish to load joints and filenames (train: %d, test: %d)',
                    self.train_data.get_size(), self.test_data.get_size())

    def _extract_joint(self, raw_coords):
        ''' Extract joint coordinates based on JOINT_MAP '''
        joint = np.empty((len(JOINT_MAP), 2), dtype=np.float32)
        for key, row in JOINT_MAP.items():
            if key == 'head':
                # head coordinate
                raw_idx0 = Flic.raw_joint_map['leye']
                raw_idx1 = Flic.raw_joint_map['reye']
                raw_idx2 = Flic.raw_joint_map['nose']
                joint[row] = (raw_coords[raw_idx0] +
                              raw_coords[raw_idx1] +
                              raw_coords[raw_idx2]) / 3
            else:
                raw_idx = Flic.raw_joint_map[key]
                joint[row] = raw_coords[raw_idx]
        return joint
