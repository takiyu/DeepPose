# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())

# Constant variables
IMG_SIZE = (220, 220)


def copy_layers(src_model, dst_model,
                names=['conv1', 'conv2', 'conv3', 'conv4', 'conv5']):
    for name in names:
        for s, d in zip(src_model[name].params(), dst_model[name].params()):
            d.data = s.data


class DeepPoseModel(chainer.Chain):

    def __init__(self, n_joint):
        super(DeepPoseModel, self).__init__(
            conv1=L.Convolution2D(3, 96, 11, stride=4, pad=0),
            conv2=L.Convolution2D(96, 256, 5, stride=1, pad=2),
            conv3=L.Convolution2D(256, 384, 3, stride=1, pad=1),
            conv4=L.Convolution2D(384, 384, 3, stride=1, pad=1),
            conv5=L.Convolution2D(384, 256, 3, stride=1, pad=1),
            fc6=L.Linear(6 * 6 * 256, 4096),
            fc7=L.Linear(4096, 512),
            fc8=L.Linear(512, n_joint * 2),
        )
        self.train = True
        self.report = True

    def __call__(self, x_img, t_joint):
        # Alexnet
        h = F.relu(self.conv1(x_img))  # conv1
        h = F.max_pooling_2d(h, 3, stride=2, pad=0)  # max1
        h = F.local_response_normalization(h)  # norm1
        h = F.relu(self.conv2(h))  # conv2
        h = F.max_pooling_2d(h, 3, stride=2, pad=0)  # max2
        h = F.local_response_normalization(h)  # norm2
        h = F.relu(self.conv3(h))  # conv3
        h = F.relu(self.conv4(h))  # conv4
        h = F.relu(self.conv5(h))  # conv5
        h = F.max_pooling_2d(h, 3, stride=2, pad=0)  # pool5

        h = F.dropout(F.relu(self.fc6(h)), ratio=0.6, train=self.train)  # fc6
        h = F.dropout(F.relu(self.fc7(h)), ratio=0.6, train=self.train)  # fc7
        h_joint = self.fc8(h)  # fc8

        # Loss
        if self.train:
            loss = F.mean_squared_error(h_joint, t_joint)

        if self.report:
            if self.train:
                # Report losses
                chainer.report({'loss': loss}, self)

            # Report results
            predict_data = {'img': x_img, 'joint': h_joint}
            teacher_data = {'img': x_img, 'joint': t_joint}
            chainer.report({'predict': predict_data}, self)
            chainer.report({'teacher': teacher_data}, self)

            # Report layer weights
            chainer.report({'conv1_w': {'weights': self.conv1.W},
                            'conv2_w': {'weights': self.conv2.W},
                            'conv3_w': {'weights': self.conv3.W},
                            'conv4_w': {'weights': self.conv4.W},
                            'conv5_w': {'weights': self.conv5.W}}, self)

        if self.train:
            return loss
        else:
            return {'img': x_img, 'joint': h_joint}
