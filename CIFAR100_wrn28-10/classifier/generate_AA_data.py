# -*- coding: utf-8 -*-
# @Time    : 1/8/21 3:16 PM
# @Author  : Pu Li
# @Email   : pli5270@sdsu.edu
# @File    : generate_AA_data.py
# @Description: Generate data by auto-augment policy

import os
import sys
this_dir = os.path.dirname(__file__)
sys.path.append(this_dir)
import numpy as np

import policies as found_policies
import augmentation_transforms
import h5py
import os

source_data_file = '/data2/data_augmentation/RL_Augment/data/cifar100/uint8/train.h5'
output_dir = '/data2/data_augmentation/RL_Augment/data/cifar100/autoaug'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

h5 = h5py.File(source_data_file)
data = h5['data'].value
label = h5['label'].value
h5.close()
output_data = data.copy()

data = data.astype(np.float32) / 255.0
mean = augmentation_transforms.MEANS
std = augmentation_transforms.STDS
data = (data - mean) / std

good_policies = found_policies.good_policies()
ndata = data.shape[0]

start_iter = 30
end_iter = 70
for i_iter in range(start_iter, end_iter):
    for i in range(ndata):
        if (i+1) % 100 == 0:
            print(i+1)
        epoch_policy = good_policies[np.random.choice(len(good_policies))]
        curr_data = data[i, ]
        curr_data = augmentation_transforms.apply_policy(epoch_policy, curr_data)
        curr_data = (curr_data * std + mean) * 255
        curr_data = curr_data.astype(np.uint8)
        output_data[i, ] = curr_data

    hf = h5py.File(output_dir + '/train_aa_%d.h5' % (i_iter), 'w')
    hf.create_dataset('data', data=output_data)
    hf.create_dataset('label', data=label)
    hf.close()
