#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import log_initializer

import cv2
from logging import getLogger, DEBUG, INFO, WARNING
import multiprocessing
import six

import imgviewer

# logging
log_initializer.setFmt()
log_initializer.setRootLevel(WARNING)
logger = getLogger(__name__)
logger.setLevel(DEBUG)
imgviewer.logger.setLevel(INFO)


if __name__ == '__main__':
    logger.info('Start')

    # open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error('Failed to open camera')
        exit()

    # start server
    viewer_queue = multiprocessing.Queue()
    imgviewer.start(viewer_queue, stop_page=True, port=5000)

    num = 0
    mode = 0
    tabs = ['default', 'gray']
    while True:
        # wait 1 sec and capture
        for i in six.moves.range(10):
            ret, img = cap.read()
            if not ret:
                continue
            cv2.waitKey(100)

        img = cv2.resize(img, (200, 200))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if mode == 0:
            # set small image
            name = 's-%d' % num
            viewer_queue.put((tabs[0], name, {'img': img, 'width': 50}))
            viewer_queue.put((tabs[1], name, {'img': gray, 'width': 50}))
        elif mode == 1:
            # set middle image
            name = 'm-%d' % num
            viewer_queue.put((tabs[0], name, {'img': img, 'width': 100}))
            viewer_queue.put((tabs[1], name, {'img': gray, 'width': 100}))
        elif mode == 2:
            # set large image with caption (without resize)
            name = 'l-%d' % num
            viewer_queue.put((tabs[0], name, {'img': img,
                                              'cap': 'caption (num %d)' % num}))
            viewer_queue.put((tabs[1], name, {'img': gray,
                                              'cap': 'caption (num %d)' % num}))
        elif mode == 3:
            if num <= 5:
                # remove images
                viewer_queue.put((tabs[0], 's-%d' % num, None))
                viewer_queue.put((tabs[0], 'm-%d' % num, None))
                viewer_queue.put((tabs[0], 'l-%d' % num, None))
                viewer_queue.put((tabs[1], 's-%d' % num, None))
                viewer_queue.put((tabs[1], 'm-%d' % num, None))
                viewer_queue.put((tabs[1], 'l-%d' % num, None))
            elif num == 6:
                # remove gray tab
                viewer_queue.put((tabs[1], None, None))
            elif num == 7:
                # remove default tab
                viewer_queue.put((tabs[0], None, None))

        num = (num + 1) % 8
        if num == 0:
            mode = (mode + 1) % 4
