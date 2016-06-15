#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import six

import log_initializer
import datasets
import drawing
import settings

# logging
from logging import getLogger, DEBUG
log_initializer.setFmt()
log_initializer.setRootLevel(DEBUG)
logger = getLogger(__name__)


if __name__ == '__main__':
    # Load flic dataset
    flic = datasets.Flic()
    flic.load(settings.FLIC_FULL_PATH, settings.FLIC_PLUS_PATH)

    # Draw randomly
    perm = np.random.permutation(flic.test_data.get_size())
    for i in six.moves.xrange(perm.shape[0]):
        # Load pose
        img, joint = flic.test_data.get_data(perm[i])
        # Show
        logger.info('Show FLIC test %dth image', perm[i])
        drawing.draw_joint(img, joint)
        cv2.imshow('img', img)
        cv2.waitKey()
