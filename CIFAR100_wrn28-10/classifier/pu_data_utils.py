# -*- coding: utf-8 -*-
# @Time    : 1/8/21 5:10 PM
# @Author  : Pu Li
# @Email   : pli5270@sdsu.edu
# @File    : pu_data_utils.py
# @Description:
import glob
import os
import re
import numpy as np
import bisect
from collections import defaultdict
import h5py

def sort_aug_file(aug_files, pattern='step_([0-9]+).h5'):
    matcher = re.compile(pattern)
    filenames = [os.path.basename(file) for file in aug_files]
    nums = [int(matcher.findall(name)[0]) for name in filenames]
    idx = np.argsort(nums)
    aug_files = [aug_files[i] for i in idx]
    return aug_files

def read_action_file(action_file, n_step=10, n_action=9):
    with open(action_file, 'r') as f:
        lines = f.readlines()
        matcher = re.compile('[0-9]+, [0-1]')
        ndata = len(lines)
        idx_dicts = [{} for _ in range(n_step)]
        idxs = [0 for _ in range(n_step)]
        n_actions = []
        n_data = 0
        for line in lines:
            result = matcher.findall(line)
            assert len(result) == n_step
            flag = True
            for step in range(n_step):
                action_str = result[step]
                temp = action_str.split(',')
                action = int(temp[0])
                mag = int(temp[1])
                if action == n_action:
                    n_actions.append(step)
                    flag = False
                    break
                idx_dicts[step][n_data] = idxs[step]
                idxs[step] += 1
            if flag:
                n_actions.append(n_step)
            n_data += 1
    return n_actions, idx_dicts

def read_h5_data(h5_file, idx):
    file = h5py.File(h5_file)
    data = file['data'][idx, ]
    file.close()
    return data

def read_h5_label(h5_file, idx=None):
    file = h5py.File(h5_file)
    if idx is None:
        label = file['label'].value
    else:
        label = file['label'][idx, ]
    file.close()
    return label

def write_h5_file(h5_file, data, label):
    h5file = h5py.File(h5_file, 'w')
    h5file.create_dataset('data', data=data)
    h5file.create_dataset('label', data=label)
    h5file.close()


def prepare_h5_files(train_dir, action_files, ndata=50000, n_step=10, n_action=9):
    # datafiles = []
    split_result = train_dir.split(',')
    base_file = split_result[0]
    h5_files = [[base_file]]
    if action_files is None:
        return h5_files, None, None, None, None
    action_files = action_files.split(',')
    n_files = [[1] for _ in range(ndata)]
    h5_idx_dicts = []
    for i in range(1, len(split_result)):
        aug_files = glob.glob(os.path.join(split_result[i], '*h5'))
        aug_files = sort_aug_file(aug_files)
        h5_files.append(aug_files)
        action_file = action_files[i - 1]
        n_actions, idx_dicts = read_action_file(action_file, n_step=n_step, n_action=n_action)
        h5_idx_dicts.append(idx_dicts)
        assert len(n_actions) == ndata
        for idata in range(ndata):
            n_files[idata].append(n_actions[idata])
    n_files_total = [int(np.sum(item)) for item in n_files]
    n_files_cum = [np.cumsum(item).tolist() for item in n_files]
    return h5_files, n_files, n_files_total, n_files_cum, h5_idx_dicts

def random_sample_idx(h5_files, n_files_total, n_files_cum, h5_idx_dicts, ndata=50000):
    # rand_files = []
    # rand_files2 = []
    # rand_files3 = []
    if n_files_total is None:
        assert len(h5_files) == 1
        h5_file_idx = defaultdict(list)
        h5_file_idx_source = defaultdict(list)
        for i in range(ndata):
            h5_file_idx[h5_files[0][0]].append(i)
            h5_file_idx_source[h5_files[0][0]].append(i)
        return h5_file_idx, h5_file_idx_source
    h5_file_idx = defaultdict(list)
    h5_file_idx_source = defaultdict(list)
    for i in range(ndata):
        rand = np.random.randint(0, n_files_total[i])
        # rand_files.append(rand)
        file_list_num = bisect.bisect_right(n_files_cum[i], rand)
        pre_sum = 0 if file_list_num == 0 else n_files_cum[i][file_list_num - 1]
        file_num = rand - pre_sum
        # rand_files2.append([file_list_num, file_num])
        rand_file = h5_files[file_list_num][file_num]
        # rand_files3.append(rand_file)
        if file_list_num == 0:
            h5_file_idx[rand_file].append(i)
        else:
            h5_file_idx[rand_file].append(h5_idx_dicts[file_list_num - 1][file_num][i])
        h5_file_idx_source[rand_file].append(i)
    return h5_file_idx, h5_file_idx_source

def sample_data(h5_files, h5_file_idx, h5_file_idx_source, ndata=50000):
    data = np.zeros((ndata, 32, 32, 3)).astype(np.uint8)
    for file_list in h5_files:
        for file in file_list:
            curr_idx = h5_file_idx[file]
            curr_source_idx = h5_file_idx_source[file]
            data[curr_source_idx,] = read_h5_data(file, curr_idx)
            print('%s: %d' % (os.path.basename(file), len(h5_file_idx[file])))
    label = read_h5_label(h5_files[0][0])
    return data, label


if __name__ == '__main__':
    train_dir = '/data2/data_augmentation/RL_Augment/data/cifar100/uint8/train.h5,/data2/data_augmentation/RL_Augment/data/augmentation/20201228_1_iter1_06_train,/data2/data_augmentation/RL_Augment/data/augmentation/20201228_1_iter2_06_train'
    action_files = '/data2/data_augmentation/RL_Augment2/models/20201228_1_iter1_06/test_actions_train.log,/data2/data_augmentation/RL_Augment2/models/20201228_1_iter2_06/test_actions_train.log'

    h5_files, n_files, n_files_total, n_files_cum, h5_idx_dicts = prepare_h5_files(train_dir, action_files)
    print(n_files[:10])
    print(n_files_total[:10])
    print(n_files_cum[:10])

    # Random selection
    # print(rand_files[:10])
    # print(rand_files2[:10])
    # print(rand_files3[:10])
    h5_file_idx, h5_file_idx_source = random_sample_idx(h5_files, n_files_total, n_files_cum, h5_idx_dicts)
    data, label = sample_data(h5_files, h5_file_idx, h5_file_idx_source)
    output_h5_file = '/data2/data_augmentation/RL_Augment2/data/augmentation/20201228_1_aug0-2_train/aug.h5'
    write_h5_file(output_h5_file, data, label)