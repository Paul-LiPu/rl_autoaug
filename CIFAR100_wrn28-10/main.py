import tensorflow as tf
from dqn.agent import Agent
from dqn.environment import MyEnvironment
from config import get_config

# Parameters
flags = tf.app.flags
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')

flags.DEFINE_boolean('test_flag', False, 'Whether to do training or testing')
# flags.DEFINE_boolean('is_train', False, 'Whether to do training or testing')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
# test
flags.DEFINE_boolean('is_save', True, 'Whether to save results')
flags.DEFINE_boolean('reverse_reward', False, 'Whether to save results')
flags.DEFINE_string('save_h5', None, 'Whether to save results')

flags.DEFINE_string('dataset', 'cifar100', 'Select a dataset from mild/moderate/severe')
# flags.DEFINE_string('dataset', 'random', 'Select a dataset from mild/moderate/severe')
flags.DEFINE_string('play_model', 'models/', 'Path for testing model')
flags.DEFINE_string('test_dir', 'data/cifar100/test/', 'Path for testing model')
# training
flags.DEFINE_string('save_dir', 'models/save/', 'Path for saving models')
flags.DEFINE_integer('stop_step', 3, 'Number of steps to stop')
# flags.DEFINE_string('class_model', '/data2/data_augmentation/related_works/fast-autoaugment/models/trained/wresnet28x10_cifar_default_04.pth', 'Path for saving models')
# flags.DEFINE_string('class_model', '/data2/code/tf_models/models/research/autoaugment/training/06_model', 'Path for loading classification models')
flags.DEFINE_string('class_model', '/data2/code/tf_models/models/research/autoaugment_lipu/training/06_model2', 'Path for loading classification models')
# flags.DEFINE_string('idx_file', '/data2/data_augmentation/related_works/fast-autoaugment/models/trained/wresnet28x10_cifar_default_04_0.pkl',
#                     'Path for loading classification models')
flags.DEFINE_string('idx_file', '',
                    'Path for loading classification models')
flags.DEFINE_float('reward_mul', 10, 'Number to multiply with rewards')
# flags.DEFINE_string('log_dir', 'logs/', 'Path for logs')
FLAGS = flags.FLAGS
print('is_train: '+ str(FLAGS.is_train))
print('test_flag: '+ str(FLAGS.test_flag))
print('play_model: '+ str(FLAGS.play_model))
print('dataset: '+ str(FLAGS.dataset))
print('stop_step: '+ str(FLAGS.stop_step))


def main(_):
    with tf.Session() as sess:

        config = get_config(FLAGS)
        env = MyEnvironment(config)
        agent = Agent(config, env, sess)
        # print('No problem until hereXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

        if FLAGS.is_train:
            agent.train()
        else:
            if FLAGS.dataset == 'mine':
                agent.play_mine()
            else:
                agent.play()


if __name__ == '__main__':
    # from pytorch_models.wideresnet import WideResNet
    # from torch import load

    # classification_net = WideResNet(28, 10, dropout_rate=0.0, num_classes=100)
    # classification_net.load_state_dict(load(
    #     '/data2/data_augmentation/related_works/fast-autoaugment/models/trained/wresnet28x10_cifar100_default_04.pth')[
    #                                        'model'])
    # classification_net = classification_net.cuda()
    # classification_net.eval()

    tf.app.run()
