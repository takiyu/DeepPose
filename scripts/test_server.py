#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing
import six
import time

from convenient import start_process
import log_initializer
import datasets
from image_servers import imgviewer
import loops
import settings

# logging
from logging import getLogger, DEBUG, INFO
log_initializer.setFmt()
log_initializer.setRootLevel(DEBUG)
logger = getLogger(__name__)
imgviewer.logger.setLevel(INFO)

N_SHOW = 6

if __name__ == '__main__':
    # Load flic dataset
    flic = datasets.Flic()
    flic.load(settings.FLIC_FULL_PATH, settings.FLIC_PLUS_PATH)

    visual_que = multiprocessing.Queue()
    server_que = multiprocessing.Queue()

    # Image server process
    imgviewer.start(server_que, stop_page=False, port=settings.SERVER_PORT)

    # Visualizer loop process
    start_process(loops.visualize_pose_loop, visual_que, server_que)

    # Draw randomly
    perm = np.random.permutation(flic.test_data.get_size())
    for i in six.moves.xrange(perm.shape[0]):
        # Load pose
        img, joint = flic.test_data.get_data(perm[i])
        # Show
        logger.info('Show FLIC test %dth image', perm[i])
        visual_que.put(('pose', 'test_pose', [i % N_SHOW, [img], [joint]]))
        time.sleep(1)
