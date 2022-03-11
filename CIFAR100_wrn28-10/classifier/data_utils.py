# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Data utils for CIFAR-10 and CIFAR-100."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

this_dir = os.path.dirname(__file__)
sys.path.append(this_dir)

import copy
# import cPickle
import pickle as cPickle

import augmentation_transforms
import numpy as np
import policies as found_policies
import tensorflow as tf
import glob
import h5py
import pu_data_utils as util


# pylint:disable=logging-format-interpolation


class DataSet(object):
    """Dataset object that produces augmented training and eval data."""

    def __init__(self, hparams):
        self.hparams = hparams
        self.epochs = 0
        self.curr_train_index = 0

        all_labels = []

        self.good_policies = found_policies.good_policies()

        # Determine how many databatched to load
        num_data_batches_to_load = 5
        total_batches_to_load = num_data_batches_to_load
        train_batches_to_load = total_batches_to_load
        assert hparams.train_size + hparams.validation_size <= 50000
        if hparams.eval_test:
            total_batches_to_load += 1
        # Determine how many images we have loaded
        total_dataset_size = 10000 * num_data_batches_to_load
        train_dataset_size = total_dataset_size
        if hparams.eval_test:
            total_dataset_size += 10000

        if hparams.dataset == 'cifar10':
            all_data = np.empty((total_batches_to_load, 10000, 3072), dtype=np.uint8)
        elif hparams.dataset == 'cifar100':
            assert num_data_batches_to_load == 5
            # all_data = np.empty((1, 50000, 3072), dtype=np.uint8)
            if hparams.eval_test:
                test_data = np.empty((1, 10000, 3072), dtype=np.uint8)

        if hparams.dataset == 'cifar10':
            tf.logging.info('Cifar10')
            datafiles = []
            print(hparams.train_dir)
            h5_files, n_files, n_files_total, n_files_cum, h5_idx_dicts = util.prepare_h5_files(hparams.train_dir,
                                                                                                hparams.action_files,
                                                                                                ndata=50000,
                                                                                                n_step=hparams.n_step,
                                                                                                n_action=hparams.n_action)
            self.h5_files = h5_files
            self.n_files = n_files
            self.n_files_total = n_files_total
            self.n_files_cum = n_files_cum
            self.h5_idx_dicts = h5_idx_dicts
            if hparams.eval_test:
                datafiles.append('test_batch')
            num_classes = 10
        elif hparams.dataset == 'cifar100':
            datafiles = []
            print(hparams.train_dir)
            h5_files, n_files, n_files_total, n_files_cum, h5_idx_dicts = util.prepare_h5_files(hparams.train_dir,
                                                                                                hparams.action_files,
                                                                                                ndata=50000,
                                                                                                n_step=hparams.n_step,
                                                                                                n_action=hparams.n_action)
            self.h5_files = h5_files
            self.n_files = n_files
            self.n_files_total = n_files_total
            self.n_files_cum = n_files_cum
            self.h5_idx_dicts = h5_idx_dicts
            if hparams.eval_test:
                datafiles.append('test')
            num_classes = 100

        else:
            raise NotImplementedError('Unimplemented dataset: ', hparams.dataset)
        self.num_classes = num_classes
        if hparams.dataset != 'test':
            #total_dataset_size = 0
            #train_dataset_size = 0
            print(h5_files)

            h5_file_idx, h5_file_idx_source = util.random_sample_idx(self.h5_files, self.n_files_total,
                                                                     self.n_files_cum, self.h5_idx_dicts)
            self.h5_file_idx = h5_file_idx
            self.h5_file_idx_source = h5_file_idx_source
            data, label = util.sample_data(self.h5_files, self.h5_file_idx, self.h5_file_idx_source)

            all_data = [data]
            all_label = [label]
            for file_num, f in enumerate(datafiles):
                d = unpickle(os.path.join(hparams.data_path, f))
                test_data = copy.deepcopy(d['data'])
                test_data = test_data.reshape(-1, 3072)
                test_data = test_data.reshape(-1, 3, 32, 32)
                test_data = test_data.transpose(0, 2, 3, 1)
                total_dataset_size += test_data.shape[0]
                if hparams.dataset == 'cifar10':
                    labels = np.array(d['labels'])
                else:
                    labels = np.array(d['fine_labels'])
                all_data.append(test_data)
                all_label.append(labels)

            all_data = np.concatenate(all_data, axis=0)
            all_labels = np.concatenate(all_label, axis=0)

        all_data = all_data.astype(np.float32) / 255.0
        mean = augmentation_transforms.MEANS
        std = augmentation_transforms.STDS
        tf.logging.info('mean:{}    std: {}'.format(mean, std))

        all_data = (all_data - mean) / std
        all_labels = np.eye(num_classes)[np.array(all_labels, dtype=np.int32)]
        assert len(all_data) == len(all_labels)
        tf.logging.info(
            'In CIFAR10 loader, number of images: {}'.format(len(all_data)))

        # Break off test data
        if hparams.eval_test:
            self.test_images = all_data[train_dataset_size:]
            self.test_labels = all_labels[train_dataset_size:]

        if hasattr(hparams, 'index_file'):
            tf.logging.info(
                'Loading index file: {}'.format(hparams.index_file))
            with open(hparams.index_file, 'rb') as f:
                idx_data = cPickle.load(f)
            train_idx = idx_data['train_idx']
            valid_idx = idx_data['valid_idx']
            self.train_idx = train_idx
            self.valid_idx = valid_idx

            # Shuffle the rest of the data
            all_data = all_data[:train_dataset_size]
            all_labels = all_labels[:train_dataset_size]
            train_size, val_size = len(train_idx), len(valid_idx)
            hparams.train_size = train_size
            hparams.validation_size = val_size

            # Break into train and val
            self.train_images = all_data[train_idx,]
            self.train_labels = all_labels[train_idx,]
            self.val_images = all_data[valid_idx,]
            self.val_labels = all_labels[valid_idx,]
            self.num_train = self.train_images.shape[0]

            np.random.seed(0)
            perm = np.arange(train_size)
            np.random.shuffle(perm)
            self.train_images = self.train_images[perm]
            self.train_labels = self.train_labels[perm]

            perm = np.arange(val_size)
            np.random.shuffle(perm)
            self.val_images = self.val_images[perm]
            self.val_labels = self.val_labels[perm]
        else:
            # Shuffle the rest of the data
            all_data = all_data[:train_dataset_size]
            all_labels = all_labels[:train_dataset_size]
            np.random.seed(0)
            perm = np.arange(len(all_data))
            np.random.shuffle(perm)
            all_data = all_data[perm]
            all_labels = all_labels[perm]

            # Break into train and val
            train_size, val_size = train_dataset_size, hparams.validation_size
            self.train_images = all_data[:train_size]
            self.train_labels = all_labels[:train_size]
            # self.val_images = all_data[train_size:train_size + val_size]
            # self.val_labels = all_labels[train_size:train_size + val_size]
            self.val_images = self.test_images.copy()
            self.val_labels = self.test_labels.copy()
            self.num_train = self.train_images.shape[0]
            hparams.train_size = self.num_train
            tf.logging.info('# Data: %d, %d' % (self.num_train, total_dataset_size))

    def update_data(self):
        h5_file_idx, h5_file_idx_source = util.random_sample_idx(self.h5_files, self.n_files_total,
                                                                 self.n_files_cum, self.h5_idx_dicts)
        self.h5_file_idx = h5_file_idx
        self.h5_file_idx_source = h5_file_idx_source
        all_data, all_label = util.sample_data(self.h5_files, self.h5_file_idx, self.h5_file_idx_source)
        all_data = all_data.astype(np.float32) / 255.0
        mean = augmentation_transforms.MEANS
        std = augmentation_transforms.STDS
        all_data = (all_data - mean) / std
        all_label = np.eye(self.num_classes)[np.array(all_label, dtype=np.int32)]
        if hasattr(self, 'train_idx'):
            self.train_images = all_data[self.train_idx,]
            self.train_labels = all_label[self.train_idx,]
        else:
            self.train_images = all_data
            self.train_labels = all_label

    def next_batch(self):
        """Return the next minibatch of augmented data."""
        next_train_index = self.curr_train_index + self.hparams.batch_size
        if next_train_index > self.num_train:
            # Increase epoch number
            epoch = self.epochs + 1
            self.reset()
            self.epochs = epoch
        batched_data = (
            self.train_images[self.curr_train_index:
                              self.curr_train_index + self.hparams.batch_size],
            self.train_labels[self.curr_train_index:
                              self.curr_train_index + self.hparams.batch_size])
        final_imgs = []

        images, labels = batched_data
        # for data in images:
        #   epoch_policy = self.good_policies[np.random.choice(
        #       len(self.good_policies))]
        #   final_img = augmentation_transforms.apply_policy(
        #       epoch_policy, data)
        #   final_img = augmentation_transforms.random_flip(
        #       augmentation_transforms.zero_pad_and_crop(final_img, 4))
        #   # Apply cutout
        #   final_img = augmentation_transforms.cutout_numpy(final_img)
        #   final_imgs.append(final_img)

        for data in images:
            if self.hparams.autoaug:
                epoch_policy = self.good_policies[np.random.choice(
                    len(self.good_policies))]
                data = augmentation_transforms.apply_policy(
                    epoch_policy, data)
            final_img = augmentation_transforms.random_flip(
                augmentation_transforms.zero_pad_and_crop(data, 4))
            # Apply cutout
            if self.hparams.cutout:
                final_img = augmentation_transforms.cutout_numpy(final_img)
            final_imgs.append(final_img)
        batched_data = (np.array(final_imgs, np.float32), labels)
        self.curr_train_index += self.hparams.batch_size
        return batched_data

    def reset(self):
        """Reset training data and index into the training data."""
        self.epochs = 0
        # Shuffle the training data
        perm = np.arange(self.num_train)
        np.random.shuffle(perm)
        assert self.num_train == self.train_images.shape[
            0], 'Error incorrect shuffling mask'
        self.train_images = self.train_images[perm]
        self.train_labels = self.train_labels[perm]
        self.curr_train_index = 0


# def unpickle(f):
#   tf.logging.info('loading file: {}'.format(f))
#   fo = tf.gfile.Open(f, 'r')
#   d = cPickle.load(fo)
#   fo.close()
#   return d

def unpickle(f):
    tf.logging.info('loading file: {}'.format(f))
    fo = tf.gfile.Open(f, 'rb')
    d = cPickle.load(fo, encoding='bytes')
    result = {}
    for key in d.keys():
        if isinstance(d[key], bytes):
            d[key] = d[key].decode('utf-8')
        if isinstance(d[key], list):
            for i in range(len(d[key])):
                if isinstance(d[key][i], bytes):
                    d[key][i] = d[key][i].decode('utf-8')
        result[key.decode('utf-8')] = d[key]
    fo.close()
    return result
