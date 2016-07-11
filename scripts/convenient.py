# -*- coding: utf-8 -*-
import multiprocessing
import normalizers
import os
import os.path
import threading

import settings

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def mkdir_to_save(filename):
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_inited_pose_normalizer(train_loader, face_detector=None):
    normalizer = normalizers.FaceBasedPoseNormalizer()
    ret = normalizer.load(settings.FACIAL_NORMALIZER_PATH)  # load
    if not ret:
        # training
        if face_detector is None:
            logger.critical('To train FaceBasedPoseNormalizer, face_detector' +
                            'is needed')
            return None
        normalizer.train(train_loader, face_detector,
                         settings.N_NORMALIZER_TRAIN)
        # save
        normalizer.save(settings.FACIAL_NORMALIZER_PATH)
    return normalizer


def start_process(target, *args):
    if settings.NO_PROCESS:
        start_thread(target, *args)
    else:
        process = multiprocessing.Process(target=target, args=args)
        process.daemon = True
        process.start()


def start_thread(target, *args):
    thread = threading.Thread(target=target, args=args)
    thread.daemon = True
    thread.start()
