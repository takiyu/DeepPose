# -*- coding: utf-8 -*-
import numpy as np
import six

import alexnet
import drawing
import normalizers

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def load_pose_loop(tag, out_que, next_evt, loader, batch_size, conv_func=None,
                   random=False):
    ''' Pose data loading loop
    Arguments:
        conv_func: data convert function.
                   (img, joint, param, tag) will be passed and should
                   return (img, joint) or (imgs, joints).
    '''
    logger.info('Start loading loop (%s)', tag)

    # Loading loop
    while True:
        # Wait for request
        next_evt.wait()
        next_evt.clear()

        # One epoch
        logger.info('Start to load one epoch (%s)', tag)
        if random:
            perm = np.random.permutation(loader.get_size())
        else:
            perm = np.arange(0, loader.get_size())
        for i in six.moves.xrange(0, perm.shape[0], batch_size):
            # Load pose
            imgs, joints = loader.get_data(perm[i: i + batch_size],
                                           conv_func, tag)
            # single -> multi
            if imgs.ndim == 3:
                imgs = imgs.reshape((1,) + imgs.shape)
                joints = joints.reshape((1,) + joints.shape)
            out_que.put((imgs, joints))
            logger.debug('Put one batch (%s)', tag)

        out_que.put(None)  # end notification
        logger.debug('Finished to load one epoch (%s)', tag)


def predict_pose_loop(xp, data_que, load_evt, out_que, load_model_func,
                      joint_scale_mode):
    ''' Prediction loop '''
    logger.info('Start prediction loop')

    # Load models
    model = load_model_func()

    # Image shape
    xp_img_shape = xp.asarray([alexnet.IMG_WIDTH, alexnet.IMG_HEIGHT],
                              dtype=np.float32)

    # One epoch
    load_evt.set()  # Load request

    while True:
        # Get data
        logger.debug('Start predict batch')
        data = data_que.get()
        if data is None:
            break  # exit
        raw_imgs, _ = data  # joints are not needed

        # Numpy -> Chainer variable
        x = normalizers.conv_imgs_to_chainer(xp, raw_imgs, train=False)

        # Forward one batch
        pred = model(x)

        # Chainer variable -> Numpy
        pred_joints = normalizers.conv_joints_from_chainer(xp, pred,
                                                           xp_img_shape,
                                                           joint_scale_mode)

        out_que.put(pred_joints)

    # Exit
    out_que.put(None)
    logger.info('End of prediction loop')


def train_pose_loop(xp, n_epoch, train_data_que, train_load_evt,
                    test_data_que, test_load_evt, visual_que,
                    load_states_func, save_states_func, joint_scale_mode):
    ''' Main training loop '''
    logger.info('Start training loop')

    # Setup model, optimizer and losses
    epoch_cnt, model, optimizer, train_losses, test_losses = load_states_func()

    # Image shape
    xp_img_shape = xp.asarray([alexnet.IMG_WIDTH, alexnet.IMG_HEIGHT],
                              dtype=np.float32)

    # Main loop
    while epoch_cnt <= n_epoch:
        mode_str = 'train' if model.train else 'test'

        # One epoch
        logger.info('Start %s epoch %d', mode_str, epoch_cnt)
        if model.train:
            logger.info('Optimizer learning rate: %f', optimizer.lr)
            train_load_evt.set()  # Load request
        else:
            test_load_evt.set()  # Load request

        sum_loss = 0.0
        loss_denom = 0.0
        next_data_idx = 0
        while True:
            # Get data
            logger.debug('Start %s batch', mode_str)
            if model.train:
                data = train_data_que.get()
            else:
                data = test_data_que.get()
            if data is None:
                break  # exit
            raw_imgs, raw_joints = data

            # Numpy -> Chainer variable
            x = normalizers.conv_imgs_to_chainer(xp, raw_imgs, model.train)
            t = normalizers.conv_joints_to_chainer(xp, raw_joints,
                                                   xp_img_shape, model.train,
                                                   joint_scale_mode)

            # Forward one batch
            if model.train:
                optimizer.update(model, x, t)  # Train
            else:
                model(x, t)  # Test

            # Accumulate loss
            sum_loss += float(model.loss.data)
            loss_denom += 1.0

            # Chainer variable -> Numpy
            pred_joints = normalizers.conv_joints_from_chainer(xp, model.pred,
                                                               xp_img_shape,
                                                               joint_scale_mode)

            # Send pose data
            visual_data = [next_data_idx, raw_imgs, raw_joints, pred_joints]
            visual_que.put(('pose_comp', mode_str, visual_data))
            next_data_idx += raw_imgs.shape[0]

        # Show loss
        loss = sum_loss / loss_denom
        logger.info('%s epoch %d: loss %f', mode_str, epoch_cnt, loss)
        # Append loss
        if model.train:
            train_losses.append(loss)
        else:
            test_losses.append(loss)
        # Send loss data
        visual_que.put(('loss_graph', 'graph', [train_losses, test_losses]))

        # End of one epoch
        if not model.train:
            # Save model, optimizer and losses
            save_states_func(epoch_cnt, model, optimizer, train_losses,
                             test_losses)
            epoch_cnt += 1

        # Next mode
        model.train = not model.train

    # Exit
    visual_que.put(None)
    logger.info('End of train loop')


def visualize_pose_loop(visual_que, server_que, max_cnt=-1):
    ''' Visualizer loop

    Arguments:
        * visual_que(Queue):
            Input queue and (data_type, tab, data) should be put.
            * data_type == 'pose':
                data is [first_img_idx, imgs, joints]
            * data_type == 'pose_comp':
                data is [first_img_idx, imgs, joints, pred_joints]
            * data_type == 'loss_graph':
                data is [train_losses, test_losses]

        * server_que(Queue):
            Output queue which corresponds to image_servers.imgviewer

        * max_cnt(int):
            The maximum number of the image count.
    '''
    while True:
        # Wait for data
        raw_data = visual_que.get()
        if raw_data is None:
            break

        data_type, tab, data = raw_data

        # Pose
        if data_type == 'pose':
            # Unpack
            data_idx, imgs, joints = data
            # Draw joints and show
            for img, joint in zip(imgs, joints):
                # Check image count
                if max_cnt > 0 and data_idx >= max_cnt:
                    data_idx += 1
                    break
                # Draw and show
                drawing.draw_joint(img, joint, color_scale=1)
                img_name = 'img%05d' % data_idx
                data_idx += 1
                server_que.put((tab, img_name, {'img': img * 255}))

        # Pose compare
        elif data_type == 'pose_comp':
            # Unpack
            data_idx, imgs, joints, pred_joints = data
            # Draw joints and show
            for img, joint, pred_joint in zip(imgs, joints, pred_joints):
                # Check image count
                if max_cnt > 0 and data_idx >= max_cnt:
                    data_idx += 1
                    continue
                # Draw and show
                drawing.draw_joint(img, joint, color_scale=0.2)
                drawing.draw_joint(img, pred_joint, color_scale=1)
                img_name = 'img%05d' % data_idx
                data_idx += 1
                server_que.put((tab, img_name, {'img': img * 255}))

        # Loss graph
        elif data_type == 'loss_graph':
            # Unpack
            train_losses, test_losses = data
            # Graph
            graph_img = drawing.draw_loss_graph(train_losses, test_losses)
            if graph_img is not None:
                server_que.put((tab, 'graph', {'img': graph_img}))
