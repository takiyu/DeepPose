# -*- coding: utf-8 -*-
import cv2
import numpy as np

import datasets

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())

# matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_agg as agg
except Exception as e:
    logger.error('Failed to import matplotlib')
    logger.error('[%s] %s', str(type(e)), str(e.args))
    exit()


def _draw_line(img, pt1, pt2, color, thickness=2):
    pt1 = (int(pt1[0]), int(pt1[1]))
    pt2 = (int(pt2[0]), int(pt2[1]))
    cv2.line(img, pt1, pt2, color, int(thickness))


def _draw_circle(img, pt, color, radius=4, thickness=-1):
    pt = (int(pt[0]), int(pt[1]))
    cv2.circle(img, pt, radius, color, int(thickness))


def _draw_rect(img, rect, color, thickness=2):
    p1 = (int(rect[0]), int(rect[1]))
    p2 = (int(rect[0] + rect[2]), int(rect[1] + rect[3]))
    cv2.rectangle(img, p1, p2, color, thickness)


def _draw_cross(img, pt, color, size=4, thickness=2):
    p0 = (pt[0] - size, pt[1] - size)
    p1 = (pt[0] + size, pt[1] + size)
    p2 = (pt[0] + size, pt[1] - size)
    p3 = (pt[0] - size, pt[1] + size)
    _draw_line(img, p0, p1, color, thickness)
    _draw_line(img, p2, p3, color, thickness)


def draw_joint(img, joint, thickness=3, color_scale=1):
    jmap = datasets.JOINT_MAP
    s = color_scale

    # draw joint point difference
    if len(joint) == 1:
        # TODO fix color decision method
        if color_scale < 0.5:
            color = (0, 0, 0.5)
        else:
            color = (0, 1, 0)
        # move the origin to image center
        pt = joint[0] + np.array([img.shape[1] / 2, img.shape[0] / 2])
        # draw dot as circle
        _draw_circle(img, pt, color)
        return

    # neck
    color = (s, 0, s)
    sho_center = (joint[jmap['lsho']] + joint[jmap['rsho']]) / 2
    _draw_line(img, sho_center, joint[jmap['head']], color, thickness)

    if 'lhip' in jmap and 'rhip' in jmap:
        # hip
        color = (s, 0, s / 2)
        _draw_line(img, joint[jmap['lhip']], joint[jmap['rhip']], color,
                   thickness)

        # center
        color = (0, s, s)
        hip_center = (joint[jmap['lhip']] + joint[jmap['rhip']]) / 2
        _draw_line(img, sho_center, hip_center, color, thickness)

    # shoulder
    color = (0, s / 2, s)
    _draw_line(img, joint[jmap['lsho']], joint[jmap['rsho']], color, thickness)

    # arm
    color = (s, 0, 0)
    _draw_line(img, joint[jmap['lsho']], joint[jmap['lelb']], color, thickness)
    color = (0, s, 0)
    _draw_line(img, joint[jmap['rsho']], joint[jmap['relb']], color, thickness)
    color = (0, 0, s)
    _draw_line(img, joint[jmap['lelb']], joint[jmap['lwri']], color, thickness)
    color = (s, s, 0)
    _draw_line(img, joint[jmap['relb']], joint[jmap['rwri']], color, thickness)


def draw_loss_graph(train_loss_list, test_loss_list, train_epoch_list=None,
                    test_epoch_list=None, train_color='blue', test_color='red',
                    legend_loc='upper right', title=None):
    # Axis data
    # Losses
    train_loss = np.asarray(train_loss_list)
    test_loss = np.asarray(test_loss_list)
    # Epochs
    if train_epoch_list:
        train_epoch = np.asarray(train_epoch_list)
    else:
        train_epoch = np.arange(0, len(train_loss_list))
    if test_epoch_list:
        test_epoch = np.asarray(test_epoch_list)
    else:
        test_epoch = np.arange(0, len(test_loss_list))

    # Create new figure
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(train_epoch, train_loss, label='train', color=train_color)
    ax.plot(test_epoch, test_loss, label='test', color=test_color)

    def draw_annotate(label, x, y, color):
        ax.scatter(x, y, 20, color=color)
        ax.annotate(label, xy=(x, y), xytext=(+20, +10),
                    textcoords='offset points',
                    arrowprops={'arrowstyle': '->',
                                'connectionstyle': 'arc3,rad=.2'})

    # Show min values
    if train_loss.shape[0] > 0:
        min_idx = np.argmin(train_loss)
        x, y = train_epoch[min_idx], train_loss[min_idx]
        draw_annotate('min train loss: %0.3f' % y, x, y, train_color)
    if test_loss.shape[0] > 0:
        min_idx = np.argmin(test_loss)
        x, y = test_epoch[min_idx], test_loss[min_idx]
        draw_annotate('min test loss: %0.3f' % y, x, y, test_color)

    # Settings
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss rate")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc=legend_loc)
    if title is not None:
        ax.set_title(title)

    # Draw
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    img = np.fromstring(renderer.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))

    # Close
    plt.close('all')

    return img
