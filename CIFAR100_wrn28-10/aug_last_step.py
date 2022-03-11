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
import argparse
# from matplotlib import pyplot as plt

def read_h5_file(file):
    h5file = h5py.File(file, 'r')
    data = h5file['data'].value
    if data.dtype != np.uint8:
        data = (data * 255).astype(np.uint8)
    label = h5file['label'].value
    h5file.close()
    return data, label

def write_h5_file(file, data, label):
    h5file = h5py.File(file, 'w')
    h5file.create_dataset('data', data=data)
    h5file.create_dataset('label', data=label)
    h5file.close()


parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default='./', help='home directory')
parser.add_argument('--model_dir', type=str, default='models', help='model directory')
parser.add_argument('--ref_data', type=str, default='data/cifar100/train/train.h5', help='reference data path')
parser.add_argument('--h5_dir', type=str, default='data/augmentation', help='reference data path')
parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
parser.add_argument('--action_filename', type=str, default='test_actions_train.log', help='action output filename')
parser.add_argument('--n_step', type=int, default=10, help='maximum episode length')
parser.add_argument('--n_action', type=int, default=2, help='number of actions')
config = parser.parse_args()

h5_dir = os.path.join(config.base_dir, config.h5_dir, config.exp_name + '_train')

n_step = config.n_step
action_file = os.path.join(config.base_dir, config.model_dir, config.exp_name, config.action_filename)
ref_h5_file = os.path.join(config.base_dir, config.ref_data)
output_dir = os.path.join(h5_dir, 'last_step')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_h5 = os.path.join(output_dir, 'aug.h5')

datas = []
labels = []
for step in range(n_step):
    h5_file = os.path.join(h5_dir, 'step_' + str(step) + '.h5')
    data, label = read_h5_file(h5_file)
    print(data.dtype)
    print(data.shape)
    datas.append(data)
    labels.append(label)

ref_data, ref_label = read_h5_file(ref_h5_file)
ndata = ref_data.shape[0]
with open(action_file, 'r') as f:
    lines = f.readlines()
    # matcher = re.compile('\([0-9]+, [0-1]\)')
    matcher = re.compile('[0-9]+, [0-1]')
    ndata = len(lines)
    finished = np.zeros(ndata) == 1
    num_cum = 0
    for step in range(n_step - 1, -1, -1):
        transformed = []
        transformed2 = []
        for n, line in enumerate(lines):
            if finished[n]:
                transformed.append(False)
                transformed2.append(False)
            else:
                result = matcher.findall(line)
                assert len(result) == n_step
                action_str = result[step]
                temp = action_str.split(',')
                action = int(temp[0])
                mag = int(temp[1])
                if action == config.n_action:
                    transformed.append(False)
                else:
                    transformed.append(True)
                    transformed2.append(True)
                    finished[n] = True
        assert np.sum(transformed) == np.sum(transformed2)
        assert len(transformed) == ndata
        print('%d, %d' % (len(transformed2), datas[step].shape[0]))
        assert len(transformed2) == datas[step].shape[0]
        assert np.sum(transformed) == np.sum(transformed2)

        temp = np.sum(ref_label[transformed] == labels[step][transformed2])
        assert np.sum(ref_label[transformed] == labels[step][transformed2]) == np.sum(transformed2)
        num_cum += np.sum(transformed)
        print('step %d: %d, %d, %d' % (step, np.sum(transformed), num_cum, datas[step][transformed2, ].shape[0]))
        ref_data[transformed, ...] = datas[step][transformed2, ]
        ref_label[transformed, ...] = labels[step][transformed2, ]
write_h5_file(output_h5, ref_data, ref_label)







