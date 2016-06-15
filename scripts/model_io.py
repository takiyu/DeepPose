# -*- coding: utf-8 -*-
import numpy as np
import os

from chainer import cuda
from chainer import optimizers
from chainer import serializers

import alexnet
import convenient
import datasets
import settings

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def create_modifier(stage_cnt, joint_idx):
    if stage_cnt == 0:
        return "_s%d" % stage_cnt
    else:
        return "_s%d_j%d" % (stage_cnt, joint_idx)


def setup_model(stage_cnt):
    if stage_cnt == 0:
        model = alexnet.Alex(len(datasets.JOINT_MAP))
    else:
        model = alexnet.Alex(1)
    return model


def setup_initial_states(stage_cnt):
    ''' Setup model, optimizer and losses '''
    # Epoch count
    epoch_cnt = 0

    # Setup model
    model = setup_model(stage_cnt)
    model.train = True
    if settings.GPU >= 0:  # GPU setup
        cuda.get_device(settings.GPU).use()
        model.to_gpu()

    # Optimizer
    if stage_cnt == 0:
        optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    else:
        optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Losses
    train_losses = list()
    test_losses = list()

    return epoch_cnt, model, optimizer, train_losses, test_losses


def save_best_model(stage_cnt, joint_idx, model):
    modif = create_modifier(stage_cnt, joint_idx)
    filename = settings.BEST_MODEL % modif
    logger.info('Save best model to %s', filename)
    convenient.mkdir_to_save(filename)
    serializers.save_npz(filename, model)


def load_best_model(stage_cnt, joint_idx, gpu=False, train=False):
    modif = create_modifier(stage_cnt, joint_idx)
    filename = settings.BEST_MODEL % modif
    logger.info('Load model from %s', filename)
    if not os.path.exists(filename):
        raise FileNotFoundError
    model = setup_model(stage_cnt)
    serializers.load_npz(filename, model)
    if settings.GPU >= 0:  # GPU setup
        cuda.get_device(settings.GPU).use()
        model.to_gpu()
    model.train = train
    return model


def load_states(stage_cnt, joint_idx):
    ''' Load model, optimizer, and losses '''
    _, model, optimizer, _, _ = setup_initial_states(stage_cnt)

    modif = create_modifier(stage_cnt, joint_idx)

    # Alexnet model
    filename = settings.RESUME_MODEL % modif
    logger.info('Load model from %s', filename)
    serializers.load_npz(filename, model)
    if settings.GPU >= 0:  # GPU setup
        cuda.get_device(settings.GPU).use()
        model.to_gpu()

    # Optimizer
    optimizer.setup(model)
    filename = settings.RESUME_OPTIMIZER % modif
    logger.info('Load optimizer from %s', filename)
    serializers.load_npz(filename, optimizer)

    # Losses
    filename = settings.RESUME_LOSS % modif
    logger.info('Load loss history from %s', filename)
    loss_data = np.load(filename)
    train_losses = loss_data['train'].tolist()
    test_losses = loss_data['test'].tolist()
    assert(len(train_losses) == len(test_losses))

    # Epoch count
    epoch_cnt = len(train_losses)
    logger.info('Resume from epoch %d', epoch_cnt)

    return epoch_cnt, model, optimizer, train_losses, test_losses


def save_states(stage_cnt, joint_idx, epoch_cnt, model, optimizer, train_losses,
                test_losses):
    ''' Save model, optimizer, and losses

    If latest loss is the best, best model will be saved.
    '''

    modif = create_modifier(stage_cnt, joint_idx)

    # Save latest model
    filename = settings.RESUME_MODEL % modif
    logger.info('Save model to %s', filename)
    convenient.mkdir_to_save(filename)
    serializers.save_npz(filename, model)

    # Save latest optimizer
    filename = settings.RESUME_OPTIMIZER % modif
    logger.info('Save optimizer to %s', filename)
    convenient.mkdir_to_save(filename)
    serializers.save_npz(filename, optimizer)

    # Save latest loss history
    logger.info('Save loss history to %s', filename)
    filename = settings.RESUME_LOSS % modif
    convenient.mkdir_to_save(filename)
    np.savez(filename, train=train_losses, test=test_losses)

    # Save best model (check current loss)
    if epoch_cnt == 0 or np.min(test_losses[:-1]) > test_losses[-1]:
        save_best_model(stage_cnt, joint_idx, model)
