import tensorflow as tf
import pickle
from dqn.augmentations import augment_list
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class AgentConfig(object):
    # train / test
    is_train = False
    reverse_reward = False

    # LSTM
    # h_size = 50
    h_size = 70
    lstm_in = 32

    # test model
    play_model = 'models/'
    is_save = True

    # train model
    save_dir = 'models/save/'
    log_dir = 'logs/'
    save_h5 = None
    test_dir = None
    memory_size = 500000
    learn_start = 5000
    test_step = 50000
    save_step = 50000
    # test_step = 1000
    # save_step = 1000
    # save_step = 5000
    # save_step = 1000
    max_step = 2000000
    target_q_update_step = 10000
    # batch_size = 32
    batch_size = 64
    test_batch_size = 25
    train_frequency = 4
    discount = 0.99
    # learning rate
    learning_rate = 0.0001
    learning_rate_minimum = 0.000025
    learning_rate_decay = 0.5
    learning_rate_decay_step = 1000000
    # experience replay
    ep_start = 1.  # 1: fully random; 0: no random
    ep_end = 0.1
    ep_end_t = 1000000

    # debug
    # learn_start = 500
    # test_step = 500
    # save_step = 1000
    # target_q_update_step = 1000

class EnvironmentConfig(object):
    # params for environment
    screen_width  = 32
    screen_height = 32
    # screen_width = 63
    # screen_height = 63

    screen_channel = 3
    dataset = 'cifar100'  # cifar10 / cifar100
    # test_batch = 2048  # test how many patches at a time
    # test_batch = 10000  # test how many patches at a time
    test_batch = 50000  # test how many patches at a time
    # stop_step = 3
    stop_step = 10
    reward_func = 'step_prob_reward'
    # n_mag = 2
    # action_size = n_mag * 19 + 1
    # action_size = n_mag * 9 + 1
    # action_size = n_mag * 2 + 1
    num_class = 100

    mags = augment_list(return_mag=True)
    mags_num = [len(item) for item in mags]
    mags_cum = np.cumsum(mags_num)
    action_size = int(np.sum(mags_num)) + 1

    # data path

    # class_model = '/data2/data_augmentation/related_works/fast-autoaugment/models/trained/wresnet28x10_cifar100_default_04.pth'
    class_model = ''
    reward_mul = 10


class DQNConfig(AgentConfig, EnvironmentConfig):
    pass


def get_config(FLAGS):
    config = DQNConfig
    # TF version
    tf_version = tf.__version__.split('.')
    if int(tf_version[0]) >= 1 and int(tf_version[1]) > 4:  # TF version > 1.4
        for k in FLAGS:
            v = FLAGS[k].value
            if hasattr(config, k):
                setattr(config, k, v)
    else:
        for k, v in FLAGS.__dict__['__flags'].items():
            if hasattr(config, k):
                setattr(config, k, v)

    # conf_file = '/data2/data_augmentation/related_works/fast-autoaugment/confs/wresnet28x10_cifar_default.yaml'
    # parser2 = ConfigArgumentParser(filename=conf_file, conflict_handler='resolve')

    # classification_net = get_model(C.get()['model'], num_class(config.dataset), local_rank=-1)

    # config.reward_net = classification_net
    # print('Classification model successfully loaded from: ' + config.class_model)
    config.train_dir = 'data/' + config.dataset + '/train/'
    config.val_dir = 'data/' + config.dataset + '/test/'
    # config.test_dir = 'data/' + config.dataset + '/test/'

    if hasattr(FLAGS, 'idx_file') and FLAGS.idx_file != '':
        idx_data = unpickle(FLAGS.idx_file)
        print("Load index file from " + FLAGS.idx_file)
        setattr(config, 'train_idx', idx_data['train_idx'])
        setattr(config, 'valid_idx', idx_data['valid_idx'])

    return config
