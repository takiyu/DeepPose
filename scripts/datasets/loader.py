# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import os.path

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def mkdir_to_save(filename):
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def load_img(path):
    img = cv2.imread(path)
    if img is None:
        logger.error('Failed to load image (%s)', path)
        return None
    img = img.astype(np.float32) / 255.0
    return img


class PoseDataLoader(object):
    ''' Basic Human Pose Loader
    This class hold image paths and joints (no raw image).
    Images will be read every time.
    '''

    def __init__(self):
        self.size = 0
        self.img_paths = list()
        self.joints = list()
        self.params = list()  # optional parameters

    def set_from_raw(self, img_paths, joints, params=None):
        assert(len(img_paths) == len(joints))
        assert(params is None or len(params) == len(img_paths))
        self.size = len(img_paths)
        self.img_paths = list(img_paths)
        self.joints = list(joints)
        if params is None:
            self.params = [[]] * self.size  # param is empty list
        else:
            self.params = list(params)

    def get_size(self):
        return self.size

    def save(self, filename):
        logger.info('Save pose data to %s', filename)
        # make directory
        mkdir_to_save(filename)
        # save variables
        np_img_paths = np.asarray(self.img_paths)
        np_joints = np.asarray(self.joints)
        np_params = np.asarray(self.params)
        # save
        np.savez(filename, img_paths=np_img_paths, joints=np_joints,
                 params=np_params)

    def load(self, filename):
        logger.info('Load pose data from %s', filename)
        # load
        try:
            raw = np.load(filename)
        except:
            logger.error('Failed to load pose data')
            return False
        # get all data
        np_img_paths = raw['img_paths']
        np_joints = raw['joints']
        np_params = raw['params']
        # set
        self.set_from_raw(np_img_paths, np_joints, np_params)
        return True

    def get_data(self, indices=None, conv_func=None, tag=None):
        ''' Load certain pose data
        Arguments:
            indices: data indices (int, int iterable or None)
            conv_func: data convert function.
                       (img, joint, param [, tag]) will be passed and should
                       return (img, joint) or (imgs, joints).
        '''
        # single load mode
        def single_mode(index):
            img_path = self.img_paths[index]
            joint = self.joints[index]
            param = np.asarray(self.params[index])
            # load image
            img = load_img(img_path)
            # convert
            if conv_func is None:
                return img, joint
            else:
                if tag is None:
                    return conv_func(img, joint, param)
                else:
                    return conv_func(img, joint, param, tag)

        # multi load mode
        def multi_mode(indices):
            if indices is None:
                # all
                img_paths = np.asarray(self.img_paths)
                joints = np.asarray(self.joints)
                params = np.asarray(self.params)
            elif hasattr(indices, '__iter__'):
                # some
                img_paths = np.asarray(self.img_paths)[indices]
                joints = np.asarray(self.joints)[indices]
                params = np.asarray(self.params)[indices]

            # load images to np.ndarray
            img_list = list()
            for img_path in img_paths:
                # load image
                img = load_img(img_path)
                img_list.append(img)
            imgs = np.asarray(img_list)  # not only np.float32

            if conv_func is None:
                return imgs, joints

            # convert
            new_img_list = list()
            new_joint_list = list()
            for img, joint, param in zip(imgs, joints, params):
                if tag is None:
                    new_img, new_joint = conv_func(img, joint, param)
                else:
                    new_img, new_joint = conv_func(img, joint, param, tag)
                # DEBUG: show image
                # cv2.imshow('img', new_img)
                # cv2.waitKey()
                # append
                if isinstance(new_img, list):
                    new_img_list.extend(new_img)
                    new_joint_list.extend(new_joint)
                else:
                    new_img_list.append(new_img)
                    new_joint_list.append(new_joint)
            np_imgs = np.asarray(new_img_list, dtype=np.float32)
            np_joints = np.asarray(new_joint_list, dtype=np.float32)
            return np_imgs, np_joints

        # select single or multi
        try:
            index = int(indices)
            return single_mode(index)
        except:
            return multi_mode(indices)

    def limit_size(self, limit, randomly=True):
        # limitation
        if limit <= 0:
            logger.error('Invalid limit size')
        elif 1 <= limit < self.size:
            logger.info('Limit data size (%d -> %d)', self.size, limit)
            if randomly:
                perm = np.random.permutation(self.size)
                img_paths = np.asarray(self.img_paths)[perm[:limit]]
                joints = np.asarray(self.joints)[perm[:limit]]
                params = np.asarray(self.params)[perm[:limit]]
            else:
                img_paths = np.asarray(self.img_paths)[:limit]
                joints = np.asarray(self.joints)[:limit]
                params = np.asarray(self.params)[:limit]
            self.set_from_raw(img_paths, joints, params)
        else:
            raise NotImplementedError()
