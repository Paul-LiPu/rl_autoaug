import math
import numpy as np
import cv2


import tensorflow as tf
from dqn.Autoaug.train_lib import CifarModel

hparams = tf.contrib.training.HParams(
      train_size=50000,
      validation_size=0,
      eval_test=1,
      dataset='cifar100',
      data_path='',
      batch_size=128,
      gradient_clipping_by_global_norm=5.0)
hparams.add_hparam('model_name', 'wrn')
hparams.add_hparam('num_epochs', 200)
hparams.add_hparam('wrn_size', 160)
hparams.add_hparam('lr', 0.1)
hparams.add_hparam('weight_decay_rate', 5e-4)
hparams.add_hparam('index_file', '/data2/data_augmentation/related_works/fast-autoaugment/models/trained/wresnet28x10_cifar100_default_04_0.pkl')
hparams.add_hparam('class_model', '/data2/code/tf_models/models/research/autoaugment/training/06_model')

with tf.compat.v1.variable_scope('model', reuse=False, use_resource=False):
    meval = CifarModel(hparams)
    meval.build('eval')

print('No problem until hereXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
session = tf.compat.v1.Session('', config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False))
session.run(meval.init)

checkpoint_path = tf.train.latest_checkpoint(hparams.class_model)
if checkpoint_path is not None:
    meval.saver.restore(session, checkpoint_path)

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
    image = image[np.newaxis, ]
    image = (image - MEANS) / STDS
    return image

def prob_cal(im_input, im_label):
    im_input = prepare_data(im_input)
    predictions = session.run(
        meval.predictions,
        feed_dict={
            meval.images: im_input,
            meval.labels: im_label,
        })
    pred_label = np.argmax(predictions, axis=1)
    return float(predictions[:, im_label]), int(pred_label)

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