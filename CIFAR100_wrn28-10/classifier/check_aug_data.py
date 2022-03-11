# -*- coding: utf-8 -*-
# @Time    : 12/9/20 6:50 PM
# @Author  : Pu Li
# @Email   : pli5270@sdsu.edu
# @File    : complete_data.py
# @Description:

import os
import glob
import h5py
import re
import numpy as np
import cv2
# from matplotlib import pyplot as plt

# def read_h5_file(file):
#     h5file = h5py.File(file, 'r')
#     data = h5file['data'].value
#     if data.dtype != np.uint8:
#         data = (data * 255).astype(np.uint8)
#     label = h5file['label'].value
#     h5file.close()
#     return data, label

def read_h5_file(file):
    h5file = h5py.File(file, 'r')
    data = h5file['data'].value
    label = h5file['label'].value
    h5file.close()
    return data, label

def write_h5_file(file, data, label):
    h5file = h5py.File(file, 'w')
    h5file.create_dataset('data', data=data)
    h5file.create_dataset('label', data=label)
    h5file.close()

n_step = 10
ref_h5_file = '/data2/data_augmentation/related_works/RL-Restore_aug2/data/cifar100/uint8/train.h5'
data_dir = '/data2/data_augmentation/RL_Augment/data/cifar100/autoaug'
data_h5 = os.path.join(data_dir, 'train_aa.h5')
output_dir = os.path.join(data_dir, 'imgs')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data, label = read_h5_file(data_h5)
ref_data, ref_label = read_h5_file(ref_h5_file)

output_data = np.concatenate([ref_data, data], axis=2)
output_data = output_data[:, :, :, ::-1]
for i in range(100):
    cv2.imwrite(os.path.join(output_dir, '%d.jpg' % (i)), output_data[i, ])

