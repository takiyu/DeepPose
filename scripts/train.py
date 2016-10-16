#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import training
from chainer import iterators
from chainer.training import extensions
from chainer.links.caffe import CaffeFunction

import argparse

import log_initializer

import config
import datasets
from extensions import ImgViewerExtention
from extensions import SequentialEvaluator
from imgviewer_conversions import pose_img_func, weights_img_func
from imgviewer_conversions import lossgraph_entry_func, lossgraph_img_func
import models
from models import copy_layers
import common

# logging
from logging import getLogger, DEBUG
log_initializer.setFmt()
log_initializer.setRootLevel(DEBUG)
logger = getLogger(__name__)

# Disable type check in chainer
# os.environ["CHAINER_TYPE_CHECK"] = "0"


if __name__ == '__main__':
    # Argument
    parser = argparse.ArgumentParser(description='DeepPose training script')
    parser.add_argument('--config', '-c', default='config.json',
                        help='Load config from given json file')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    args = parser.parse_args()

    logger.info('DeepPose Training')

    # Load config
    config.load(args.config)

    # Setup face detector
    face_detector = common.CascadeFaceDetector(config.facial_cascade_paths)
    # Setup FLIC dataset
    train, test = datasets.setup_flic(config.flic_cache_path, face_detector,
                                      config.flic_full_path,
                                      config.flic_plus_path)
    # Setup pose normalizer
    pose_normalizer = common.setup_pose_normalizer(config.normalizer_cache_path,
                                                   train)
    # Set face_detector
    logger.info('Set pose normalizer to FLIC loader')
    train.set_normalizer(pose_normalizer)
    test.set_normalizer(pose_normalizer)

    # Define a model
    logger.info('Define a DeepPose model')
    model = models.DeepPoseModel(len(datasets.JOINT_MAP))

    # Initialize model
    if not args.resume:
        if config.alexnet_caffemodel_path:
            # Initialize using caffemodel
            logger.info('Overwrite conv layers using caffemodel "{}"'
                        .format(config.alexnet_caffemodel_path))
            caffe_model = CaffeFunction(config.alexnet_caffemodel_path)
            copy_layers(caffe_model, model)

    # Setup GPU
    if config.gpu >= 0:
        chainer.cuda.check_cuda_available()
        chainer.cuda.get_device(config.gpu).use()
        model.to_gpu()

    # Setup an optimizer
    logger.info('Setup an optimizer')
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Setup iterators
    # (test_iter needs repeat because of SequentialEvaluator())
    logger.info('Setup train and test iterators (n_loaders: {}, {})'
                .format(config.n_loaders_train, config.n_loaders_test))
    train_iter = iterators.MultiprocessIterator(train, config.batchsize, True,
                                                True, config.n_loaders_train)
    test_iter = iterators.MultiprocessIterator(test, config.batchsize, True,
                                               True, config.n_loaders_test)

    # Setup a updater
    logger.info('Setup an updater (GPU: {})'.format(config.gpu))
    updater = training.StandardUpdater(
        train_iter, optimizer, device=config.gpu)
    # Setup a trainer
    logger.info('Setup a trainer (output directory: {})'.format(config.outdir))
    trainer = training.Trainer(updater, (config.n_epoch, 'epoch'),
                               out=config.outdir)

    # Evaluation model with shared parameters
    logger.info('Create a copy of the model for evaluation')
    eval_model = model.copy()
    eval_model.train = False

    # Extension intervals
    n_iteration = max(len(train) // config.batchsize, 1)
    test_interval = (max(len(train) // len(test), 1), 'iteration')
    save_interval = (5, 'epoch')
    log_interval = (max(n_iteration // 1, 1), 'iteration')
    progressbar_interval = 3
    imgview_pose_interval = (5, 'iteration')
    imgview_weight_interval = (1, 'epoch')
    logger.info('Test interval : {}'.format(test_interval))
    logger.info('Save interval : {}'.format(save_interval))
    logger.info('Log interval :  {}'.format(log_interval))
    logger.info('ProgressBar interval :  {}'.format(progressbar_interval))
    logger.info('ImgView pose interval :   {}'.format(imgview_pose_interval))
    logger.info('ImgView weight interval : {}'.format(imgview_weight_interval))

    # Extensions
    trainer.extend(extensions.dump_graph('main/loss'), trigger=save_interval)
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}'), trigger=save_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}'), trigger=save_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss']),
        trigger=log_interval, invoke_before_training=True)
    trainer.extend(extensions.ProgressBar(
        update_interval=progressbar_interval))

    # My extensions
    # Sequential Evaluator
    trainer.extend(
        SequentialEvaluator(test_iter, eval_model, device=config.gpu),
        trigger=test_interval)  # Sequential evaluation for imgviewer in test
    # Image Viewer for poses
    trainer.extend(ImgViewerExtention(
        ['main/predict', 'main/teacher', 'validation/main/predict',
         'validation/main/teacher'], n_imgs=[20, 20, 10, 10],
        port=config.port_pose, image_func=pose_img_func),
        trigger=imgview_pose_interval)
    # Image Viewer for weights
    trainer.extend(ImgViewerExtention(
        ['main/conv1_w', 'main/conv2_w', 'main/conv3_w', 'main/conv4_w',
         'main/conv5_w', ], n_imgs=[96, 0, 0, 0, 0],
        port=config.port_weight, image_func=weights_img_func),
        trigger=imgview_weight_interval)
    # Image Viewer for loss graph
    trainer.extend(ImgViewerExtention(
        ['lossgraph'], n_imgs=[1],
        port=config.port_lossgraph, entry_func=lossgraph_entry_func,
        image_func=lossgraph_img_func), trigger=log_interval,
        invoke_before_training=True)

    # Resume
    if args.resume:
        logger.info('Resume from "{}"'.format(args.resume))
        chainer.serializers.load_npz(args.resume, trainer)

    # Run
    logger.info('Start training')
    trainer.run()
