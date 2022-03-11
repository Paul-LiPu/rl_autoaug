# -*- coding: utf-8 -*-
# @Time    : 12/25/20 4:57 PM
# @Author  : Pu Li
# @Email   : pli5270@sdsu.edu
# @File    : work_flow.py
# @Description:
from dqn.augmentations import augment_list
import os
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
import numpy as np

curr_dir = os.path.dirname(__file__)

# Classifier training parameters
bakup_dir = 'autoaug2/run_bak'
check_dir(bakup_dir)
checkpoint_dir = 'autoaug2/training'
log_dir = 'autoaug2/logs'
data_path = 'data/cifar10/binary'
dataset = 'cifar10'
# cutout = False
cutout = True
autoaug = False

orig_data = 'data/cifar10/uint8/train.h5'
index_file = 'data/cifar10/wresnet28x10_cifar_default_04_0.pkl'
aug_data_dir = 'data/augmentation'
use_index_postfix = '_06'


# Policy network training
is_train = True
policy_log_dir = 'logs/'
policy_bak_dir = 'run_bak/'
stop_step = 10

mags = augment_list(return_mag=True)
n_action = len(mags)
save_dir = 'models'
reverse_reward = True
reward_mul = 10
# class_model = '/data2/code/tf_models/models/research/autoaugment_lipu/training/Autoaug_06_model/model.ckpt-199'
# class_model = '/data2/code/tf_models/models/research/autoaugment_lipu/training/06_model2/model.ckpt-199'
class_model_dir = checkpoint_dir
test_data_dir = 'data/cifar10'
dataset_name = 'cifar10'

check_dir(policy_bak_dir)
check_dir(policy_log_dir)
check_dir(save_dir)


### Training of WRN-28-10
def train_classifier(name, iteration, use_idx, model):
    exp_name = '%s_%s_iter%d' % (name, model, iteration)
    if use_idx:
        exp_name += use_index_postfix
    bak_file = os.path.join(bakup_dir, exp_name + '.sh')
    aug_data = [orig_data]
    action_files = []
    for i in range(0, iteration):
        exp_name_prev = '%s_iter%d%s' % (name, i, use_index_postfix)
        aug_data.append(os.path.join(aug_data_dir, '%s_train' % (exp_name_prev)))
        action_files.append(os.path.join(save_dir, '%s' % (exp_name_prev), 'test_actions_train.log'))
    action_files = ','.join(action_files)
    train_dir = ','.join(aug_data)
    index_file_str = ' \\\n               --index_file=%s' % (index_file) if use_idx else ''
    action_files_str = ' \\\n               --action_files=%s' % (action_files) if len(action_files) > 0 else ''
    command = 'python autoaug2/train_cifar.py --exp_name=%s \\\n \
              --model_name=%s \\\n \
              --checkpoint_dir=%s \\\n \
              --logdir=%s \\\n \
              --data_path=%s \\\n \
              --dataset=%s \\\n \
              --cutout=%s \\\n \
              --autoaug=%s \\\n \
              --use_cpu=0 \\\n \
              --n_action=%d \\\n \
              --n_step=%d%s \\\n \
              --train_dir=%s%s' % (exp_name, model, checkpoint_dir, log_dir, data_path, dataset, str(cutout), str(autoaug),
                                   n_action, stop_step, action_files_str, train_dir, index_file_str)
    print(command)
    with open(bak_file, 'w') as f:
        f.write(command)
    os.system(command)


# train_classifier(exp_name, 1, use_idx=True)



def train_policy(name, iteration, use_idx):
    exp_name = '%s_iter%d' % (name, iteration)
    if use_idx:
        exp_name += use_index_postfix
    # if iteration == 0:
    #     exp_name = name
    #     curr_class_model = class_model
    # else:
    curr_class_model = os.path.join(class_model_dir, '%s/model.ckpt-199' % exp_name)
    bak_file = os.path.join(policy_bak_dir, exp_name + '_policy.sh')
    # index_file_str = ' \\\n               --idx_file=%s' % (index_file) if use_idx else ''
    index_file_str = ' --idx_file=%s' % (index_file) if use_idx else ''
    # command = 'python main.py --is_train=%s \\\n \
    #           --log_dir=%s \\\n \
    #           --stop_step=%s \\\n \
    #           --save_dir=%s \\\n \
    #           --reverse_reward=%s \\\n \
    #           --class_model=%s%s' % (str(is_train), policy_log_dir, str(stop_step), save_dir, str(reverse_reward), curr_class_model, index_file_str)
    exp_save_dir = os.path.join(save_dir, exp_name)
    log_output_dir = os.path.join(policy_log_dir, exp_name)
    check_dir(log_output_dir)
    log_output_file = os.path.join(log_output_dir, exp_name + '.log')
    command = 'python main.py --is_train=%s --log_dir=%s --stop_step=%s --save_dir=%s/ --reverse_reward=%s --reward_mul=%s --class_model=%s%s > %s' % (
    str(is_train), policy_log_dir, str(stop_step), exp_save_dir, str(reverse_reward), str(reward_mul), curr_class_model, index_file_str, log_output_file)
    # command2 = 'mkdir %s' % os.path.join(policy_log_dir, exp_name)
    command2 = 'cp %s %s' % (bak_file, 'run.sh')
    command3 = 'mv %s/events.out* %s' % (policy_log_dir, os.path.join(policy_log_dir, exp_name))
    print(command)
    print(command3)
    with open(bak_file, 'w') as f:
        f.write(command + '\n')
        # f.write(command3 + '\n')

    os.system(command2)
    os.system('bash run.sh')
    os.system(command3)


def test_policy(name, iteration, use_idx, train_file=True):
    exp_name = '%s_iter%d' % (name, iteration)
    if use_idx:
        exp_name += use_index_postfix
    curr_class_model = os.path.join(class_model_dir, '%s/model.ckpt-199' % exp_name)
    test_name = 'train' if train_file else 'test'
    bak_file = os.path.join(policy_bak_dir, exp_name + '_policy_test_%s.sh' % (test_name))
    dataset_dir = os.path.join(test_data_dir, test_name)
    play_model = os.path.join(save_dir, exp_name)
    save_h5 = os.path.join(aug_data_dir, '%s_%s' % (exp_name, test_name))
    command = 'python main.py --is_train=%s --dataset=%s --play_model=%s/ --stop_step=%s --save_h5=%s --reverse_reward=%s --test_dir=%s/ --class_model=%s  --reward_mul=%s | tee %s/test_output_%s.log' % (
    str(False), dataset_name, play_model, str(stop_step), save_h5, str(reverse_reward), dataset_dir, curr_class_model, str(reward_mul), play_model, test_name)

    # command2 = 'mkdir %s' % os.path.join(policy_log_dir, exp_name)
    check_dir(os.path.join(policy_log_dir, exp_name))
    command2 = 'cp %s %s' % (bak_file, 'test.sh')
    command3 = 'mv %s/test_actions.log %s/test_actions_%s.log' % (play_model, play_model, test_name)
    command4 = 'mv results/%s results/%s_%s_%s' % (dataset_name, dataset_name, exp_name, test_name)
    print(command)
    print(command3)
    print(command4)
    with open(bak_file, 'w') as f:
        f.write(command + '\n')
        # f.write(command3 + '\n')

    os.system(command2)
    os.system('bash test.sh')
    os.system(command3)
    os.system(command4)

def generate_aug_data(name, iteration, use_idx):
    exp_name = '%s_iter%d' % (name, iteration)
    if use_idx:
        exp_name += use_index_postfix
    test_name = 'train'
    # save_h5 = os.path.join(aug_data_dir, '%s_%s' % (exp_name, test_name))
    command = 'python aug_last_step.py --exp_name %s --n_action %d' % (exp_name, n_action)
    print(command)
    bak_file = os.path.join(policy_bak_dir, exp_name + '_policy_merge.sh')
    with open(bak_file, 'w') as f:
        f.write(command + '\n')
    os.system(command)


if __name__ == '__main__':
    exp_name = '20210221'
    net_archs = [
        # 'wrn',
        # 'wrn-28-2',
        # 'wrn-40-2',
        # 'shake_shake_32',
        # 'shake_shake_96',
        'shake_shake_112',
        # 'pyramid_net',
    ]
    # iters = [5, 3, 1]
    iters = [2]
    use_idx = True
    for net_arch in net_archs:
        for i_iter in iters:
            train_classifier(exp_name, i_iter, use_idx=False, model=net_arch)
