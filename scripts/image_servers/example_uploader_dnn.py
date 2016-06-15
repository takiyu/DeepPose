#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import log_initializer

import cv2
from logging import getLogger, DEBUG, INFO, WARNING
import multiprocessing
try:
    import Queue
except:
    import queue as Queue

import imguploader
import dnn

# logging
log_initializer.setFmt()
log_initializer.setRootLevel(WARNING)
logger = getLogger(__name__)
logger.setLevel(DEBUG)
imguploader.logger.setLevel(INFO)
dnn.logger.setLevel(INFO)


if __name__ == '__main__':
    logger.info('Start')

    dnn.init()

    request_queue = multiprocessing.Queue()
    response_queue = multiprocessing.Queue()

    imguploader.start(request_queue, response_queue, stop_page=True, port=5000)

    while True:
        try:
            img = request_queue.get(block=False)

            cv2.imshow('img', img)
            res_message = dnn.predict(img)

            response_queue.put({'img': img,
                                'img_options': {'region': True},
                                'msg': res_message})
        except Queue.Empty:
            pass

        cv2.waitKey(10)
