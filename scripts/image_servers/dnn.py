# -*- coding: utf-8 -*-

from chainer import Variable
from chainer.links.caffe import CaffeFunction

import numpy as np
import cv2

import pickle
import os.path

# logging
from logging import getLogger, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)


CAFFE_MODEL = './dnn_models/bvlc_googlenet.caffemodel'
LABEL_TEXT = './dnn_models/synset_words.txt'


categories = None
model = None


def init():
    global categories, model
    categories = np.loadtxt(LABEL_TEXT, str, delimiter="\n")
    model = load_caffemodel(CAFFE_MODEL, dump=True)


def load_caffemodel(org_path, dump=True):
    pkl_path = org_path + '.pkl'
    if os.path.exists(pkl_path):
        logger.info('Load pkl model: %s' % pkl_path)
        model = pickle.load(open(pkl_path, 'rb'))
    else:
        if not os.path.exists(org_path):
            logger.error('Failed to load caffe model: %s' % org_path)
            return None
        logger.info('Load caffe model: %s' % org_path)
        model = CaffeFunction(org_path)
        if dump:
            logger.info('Save pkl model: %s' % pkl_path)
            pickle.dump(model, open(pkl_path, 'wb'))
    return model


def predict(img):
    global categories, model
    if categories is None or model is None:
        init()

    img_size = (224, 224)
    img = cv2.resize(img, img_size)
    input_data = img.transpose((2, 0, 1)).reshape(*((1, 3) + img_size))
    input_data = input_data.astype(np.float32)

    mean_image = np.empty_like(input_data)
    mean_image[:, 0] = 104
    mean_image[:, 1] = 117
    mean_image[:, 2] = 123

    input_data -= mean_image

    x = Variable(input_data)
    y, = model(inputs={'data': x}, outputs=['loss3/classifier'])

    result = zip(y.data[0].tolist(), categories)
    result.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)
    for rank, (score, name) in enumerate(result[:10], start=1):
        print('%d: %s (%f)' % (rank, name, score))

    score, name = result[0]
    return '%s (%f)' % (name, score)
