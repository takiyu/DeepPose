import chainer
import chainer.functions as F
import chainer.links as L

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


IMG_WIDTH, IMG_HEIGHT = 220, 220


class Alex(chainer.Chain):
    insize = 220

    def __init__(self, n_joint):
        super(Alex, self).__init__(
            conv1=L.Convolution2D(3, 96, 11, stride=4),
            conv2=L.Convolution2D(96, 256, 5, pad=2),
            conv3=L.Convolution2D(256, 384, 3, pad=1),
            conv4=L.Convolution2D(384, 384, 3, pad=1),
            conv5=L.Convolution2D(384, 256, 3, pad=1),
            fc6=L.Linear(9216, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, n_joint * 2),
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.pred = None

    def __call__(self, x, t=None):
        # check argument
        if t is None and self.train:
            logger.error('Teacher data is needed')
            return None

        self.clear()

        # start forwarding
        h = self.conv1(x)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.conv2(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 3, stride=2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), ratio=0.6, train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), ratio=0.6, train=self.train)
        self.pred = self.fc8(h)

        if t is not None:
            self.loss = F.mean_squared_error(self.pred, t)

        if self.train:
            return self.loss
        else:
            return self.pred
