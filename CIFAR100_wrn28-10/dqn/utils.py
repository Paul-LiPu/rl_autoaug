import math
import numpy as np
import cv2
import tensorflow as tf
# from dqn.Autoaug.train_lib import CifarModel
# flags = tf.app.flags
# FLAGS = flags.FLAGS


class Eval_engine():
    def __init__(self, config):
        hparams = tf.contrib.training.HParams(
              train_size=50000,
              validation_size=0,
              eval_test=1,
              dataset='cifar10',
              data_path='',
              batch_size=25,
              gradient_clipping_by_global_norm=5.0)
        hparams.add_hparam('model_name', 'wrn')
        hparams.add_hparam('num_epochs', 200)
        hparams.add_hparam('wrn_size', 160)
        hparams.add_hparam('lr', 0.1)
        hparams.add_hparam('weight_decay_rate', 5e-4)
        # hparams.add_hparam('index_file', '/data2/data_augmentation/related_works/fast-autoaugment/models/trained/wresnet28x10_cifar100_default_04_0.pkl')
        # hparams.add_hparam('class_model', '/data2/code/tf_models/models/research/autoaugment_lipu/training/06_model')
        # hparams.add_hparam('class_model', config.class_model)
        hparams.add_hparam('class_model', '/data2/code/tf_models/models/research/autoaugment_lipu/training/vanilla')
        # print(config.class_model)
        hparams.add_hparam('nclass', 10)

        self.hparams = hparams


        # session = tf.compat.v1.Session('', config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
        #                                                                    log_device_placement=False))
        # session = tf.get_default_session()
        # session.run(meval.init)
        # self.session = session
        #
        # checkpoint_path = tf.train.latest_checkpoint(hparams.class_model)
        # # if checkpoint_path is not None:
        # self.saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
        # self.saver.restore(session, checkpoint_path)
        # self.meval = meval
        # print(hparams.class_model)
        # checkpoint_path = tf.train.latest_checkpoint(hparams.class_model)

        # class_model = '/data2/code/tf_models/models/research/autoaugment_lipu/training/vanilla'
        # checkpoint_path = tf.train.latest_checkpoint(class_model)
        # checkpoint_path = '/data2/code/tf_models/models/research/autoaugment_lipu/training/vanilla/model.ckpt-199'
        checkpoint_path = config.class_model
        g = tf.Graph()
        with g.as_default():
            # load graph
            saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
            # input data
            # self.input_data = g.get_tensor_by_name('model/Placeholder:0')
            # self.input_label = g.get_tensor_by_name('model/Placeholder_1:0')
            # self.output_data = g.get_tensor_by_name('model/unit_last/FC/xw_plus_b:0')

            self.input_data = g.get_tensor_by_name('model_1/Placeholder:0')
            self.input_label = g.get_tensor_by_name('model_1/Placeholder_1:0')
            self.output_data = g.get_tensor_by_name('model_1/unit_last/FC/xw_plus_b:0')

        # session = tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=True))
        # session = tf.Session(graph=g, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        # session = tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=True))
        session = tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=False))
        with g.as_default():
            with session.as_default():
                saver.restore(session, checkpoint_path)
                # with tf.compat.v1.variable_scope('model', reuse=tf.AUTO_REUSE, use_resource=False):
                #     meval = CifarModel(hparams)
                #     meval.build('eval')
        self.g = g
        self.session = session


    # def prob_cal(self, im_input, im_label):
    #     data = prepare_data(im_input)
    #     # label = np.zeros((1, self.hparams.nclass)).astype(np.int32)
    #     # label[0, int(im_label)] = 1
    #     # label = np.eye(self.hparams.nclass)[np.array([im_label], dtype=np.int32)]
    #     label = np.eye(self.hparams.nclass)[np.array(im_label, dtype=np.int32)]
    #     data, label = add_dim(data, label, self.hparams.batch_size // data.shape[0])
    #     # predictions = self.session.run(
    #     #     self.meval.predictions,
    #     #     feed_dict={
    #     #         self.meval.images: im_input,
    #     #         self.meval.labels: label,
    #     #     })
    #     with self.g.as_default():
    #         with self.session.as_default():
    #             predictions = self.session.run(
    #                 self.output_data,
    #                 feed_dict={
    #                     self.input_data: data,
    #                     self.input_label: label,
    #                 })
    #             predictions = predictions[0:1, ...]
    #             pred_label = np.argmax(predictions, axis=1)
    #             # predictions_softmax = np.exp(predictions)
    #             # predictions_softmax = predictions_softmax / predictions_softmax.sum()
    #             # print((float(predictions_softmax[0, int(pred_label)]), float(predictions_softmax[0, int(im_label)]), int(pred_label), int(im_label)))
    #             # print((float(predictions[0, int(pred_label)]), float(predictions[0, int(im_label)]),
    #             #        int(pred_label), int(im_label)))
    #     # return float(predictions_softmax[0, int(im_label)]), int(pred_label)
    #     return float(predictions[0, int(im_label)]), int(pred_label)
    #     # return predictions[0, im_label], int(pred_label)
    #     # return 0.0, 0



    def prob_cal(self, im_input, im_label, k=1):
        data = prepare_data(im_input)
        # label = np.eye(self.hparams.nclass)[np.array(im_label, dtype=np.int32)]
        label = np.eye(self.hparams.nclass)[np.array(im_label, dtype=np.int32)]
        with self.g.as_default():
            with self.session.as_default():
                predictions = self.session.run(
                    self.output_data,
                    feed_dict={
                        self.input_data: data,
                        self.input_label: label,
                    })
                predictions = predictions[0:1, ...]
                pred_label = np.argmax(predictions, axis=1)
                predictions_softmax = np.exp(predictions)
                predictions_softmax = predictions_softmax / float(predictions_softmax.sum())
                # print((float(predictions[0, int(pred_label)]), float(predictions[0, int(im_label)]),
                #                    int(pred_label), int(im_label)))
        return k * float(predictions_softmax[0, int(im_label)]), int(pred_label)


    def prob_cal_batch(self, im_input, im_label, k=1):
        data = prepare_data(im_input)
        label = np.eye(self.hparams.nclass)[np.array(im_label, dtype=np.int32)]

        with self.g.as_default():
            with self.session.as_default():
                predictions = self.session.run(
                    self.output_data,
                    feed_dict={
                        self.input_data: data,
                        self.input_label: label,
                    })
                pred_label = np.argmax(predictions, axis=1)
                pred_value = np.max(predictions, axis=1)
                # print(predictions.shape)
                predictions_softmax = np.exp(predictions)
                sum_value = np.sum(predictions_softmax, axis=1)
                # print(len(sum_value))
                for i in range(len(sum_value)):
                    predictions_softmax[i, :] /= sum_value[i]
                predictions_softmax_value = np.max(predictions_softmax, axis=1)
                #predictions_softmax_value = predictions_softmax[:, 0]

                #for i in range(len(sum_value)):
                #    predictions_softmax_value[i] = predictions_softmax[i, im_label[i]]
                # for i in range(data.shape[0]):
                #     print((float(predictions[i, int(pred_label[i])]), float(predictions[i, int(im_label[i])]),
                #            int(pred_label[i]), int(im_label[i])))
        # return float(predictions_softmax[0, int(im_label)]), int(pred_label)
        return k * predictions_softmax_value, pred_label

    # def prob_cal(self, im_input, im_label):
    #     return 0, 0
    #
    # def prob_cal_batch(self, im_input, im_label):
    #     return np.zeros(im_input.shape[0]), np.zeros(im_input.shape[0])


def psnr_cal(im_input, im_label):
    loss = (im_input - im_label) ** 2
    eps = 1e-10
    loss_value = loss.mean() + eps
    psnr = 10 * math.log10(1.0 / loss_value)
    return psnr


MEANS = [0.49139968, 0.48215841, 0.44653091]
STDS = [0.24703223, 0.24348513, 0.26158784]

# from torchvision.transforms import functional as F
# _CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
# def prepare_data(image):
#     tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
#     normalized_tensor = F.normalize(tensor, _CIFAR_MEAN, _CIFAR_STD, inplace=False)
#     return normalized_tensor

# def prob_cal(net, im_input, im_label):
#     with torch.no_grad():
#         preds = net(prepare_data(im_input).cuda())
#     preds = preds.cpu()
#     pred_label = torch.argmax(preds, dim=1)
#     return float(preds[:, im_label]), int(pred_label)

def prepare_data(image):
    # image = image[np.newaxis, ]
    image = (image - MEANS) / STDS
    return image

def add_dim(data, label, dim):
    data = np.repeat(data, dim, axis=0)
    label = np.repeat(label, dim, axis=0)
    return data, label


def load_imgs(list_in, list_gt, size = 63):
    assert len(list_in) == len(list_gt)
    img_num = len(list_in)
    imgs_in = np.zeros([img_num, size, size, 3])
    imgs_gt = np.zeros([img_num, size, size, 3])
    for k in range(img_num):
        imgs_in[k, ...] = cv2.imread(list_in[k]) / 255.
        imgs_gt[k, ...] = cv2.imread(list_gt[k]) / 255.
    return imgs_in, imgs_gt


def img2patch(my_img, size=63):
    height, width, _ = np.shape(my_img)
    assert height >= size and width >= size
    patches = []
    for k in range(0, height - size + 1, size):
        for m in range(0, width - size + 1, size):
            patches.append(my_img[k: k+size, m: m+size, :].copy())
    return np.array(patches)


def data_reformat(data):
    """RGB <--> BGR, swap H and W"""
    assert data.ndim == 4
    out = data[:, :, :, ::-1]
    out = np.swapaxes(out, 1, 2)
    return out


def step_psnr_reward(psnr, psnr_pre):
    reward = psnr - psnr_pre
    return reward

def step_psnr_reward2(psnr, psnr_pre):
    reward = psnr_pre - psnr
    return reward
