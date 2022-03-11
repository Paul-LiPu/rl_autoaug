import numpy as np
# import tensorflow as tf
import os
import h5py
import cv2
from .utils import Eval_engine, step_psnr_reward, step_psnr_reward2
from .augmentations import apply_augment


class MyEnvironment(object):
    def __init__(self, config):
        self.reward = 0
        self.terminal = True
        self.stop_step = config.stop_step
        self.reward_func = config.reward_func
        self.is_train = config.is_train
        self.count = 0  # count restoration step
        self.psnr, self.psnr_pre, self.psnr_init = 0., 0., 0.
        self.config = config

        self.eval_engine = Eval_engine(config)

        if self.is_train:
            # training data
            self.train_list = [config.train_dir + file for file in os.listdir(config.train_dir) if file.endswith('.h5')]
            self.train_cur = 0
            self.train_max = len(self.train_list)
            f = h5py.File(self.train_list[self.train_cur], 'r')
            self.data = f['data'].value
            self.label = f['label'].value
            f.close()
            if hasattr(config, 'train_idx') and hasattr(config, 'valid_idx'):
                self.data = self.data[config.valid_idx, ]
                self.label = self.label[config.valid_idx]

            self.data_index = 0
            self.data_len = len(self.data)
            print("# of Training data:" + str(self.data_len))

            # validation data
            print(config.val_dir + os.listdir(config.val_dir)[0])
            f = h5py.File(config.val_dir + os.listdir(config.val_dir)[0], 'r')
            self.data_test = f['data'].value[:1000, ]
            self.label_test = f['label'].value[:1000, ]
            f.close()
            self.data_all = self.data_test
            self.label_all = self.label_test
        else:
            if config.dataset in ['cifar10', 'cifar100']:
                self.test_batch = config.test_batch
                f = h5py.File(config.test_dir + os.listdir(config.test_dir)[0], 'r')
                self.data_all = f['data'].value
                self.label_all = f['label'].value
                f.close()
                # self.data_test = f['data'].value
                # self.label_test = f['label'].value
                self.test_total = self.data_all.shape[0]
                self.name_list = [str(i) for i in range(self.test_total)]
                self.test_cur = 0

                self.data_test = self.data_all[0: min(self.test_batch, self.test_total), ...]
                self.label_test = self.label_all[0: min(self.test_batch, self.test_total), ...]
            else:
                raise ValueError('Invalid dataset!')

        if self.is_train or config.dataset!='mine':
            # input prob value
            self.base_psnr = 0.
            self.base_acc = 0.
            for k in range(0, len(self.data_test), self.config.test_batch_size):
                psnr, pred_label= self.eval_engine.prob_cal_batch(self.data_test[k:k+self.config.test_batch_size, ...],
                                                            self.label_test[k:k+self.config.test_batch_size, ...],
                                                                  k=self.config.reward_mul)
                self.base_psnr += np.sum(psnr)
                self.base_acc += np.sum(pred_label == self.label_test[k:k+config.test_batch_size, ...])
            # for k in range(0, len(self.data_test)):
            #     psnr, pred_label= self.eval_engine.prob_cal(self.data_test[k:k+1, ...],
            #                                                 self.label_test[k:k+1, ...])
            #     self.base_psnr += psnr
            #     self.base_acc += 1 if pred_label == int(self.label_test[k:k+1, ...]) else 0
            self.base_psnr /= len(self.data_test)
            self.base_acc /= len(self.data_test)
            print_str = 'base_psnr: %.4f' + ' base_acc: %.4f'
            print(print_str % (self.base_psnr, self.base_acc))

            # reward functions
            if config.reverse_reward:
                self.rewards = {'step_prob_reward': step_psnr_reward2}
            else:
                self.rewards = {'step_prob_reward': step_psnr_reward}

            self.reward_function = self.rewards[self.reward_func]

        # build toolbox
        self.action_size = config.action_size
        # toolbox_path = 'toolbox/'
        # self.graphs = []
        # self.sessions = []
        # self.inputs = []
        # self.outputs = []
        # for idx in range(12):
        #     g = tf.Graph()
        #     with g.as_default():
        #         # load graph
        #         saver = tf.train.import_meta_graph(toolbox_path + 'tool%02d' % (idx + 1) + '.meta')
        #         # input data
        #         input_data = g.get_tensor_by_name('Placeholder:0')
        #         self.inputs.append(input_data)
        #         # get the output
        #         output_data = g.get_tensor_by_name('sum:0')
        #         self.outputs.append(output_data)
        #         # save graph
        #         self.graphs.append(g)
        #     sess = tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=True))
        #     with g.as_default():
        #         with sess.as_default():
        #             saver.restore(sess, toolbox_path + 'tool%02d' % (idx + 1))
        #             self.sessions.append(sess)


    def new_image(self):
        self.terminal = False
        while self.data_index < self.data_len:
            self.img = self.data[self.data_index: self.data_index + 1, ...]
            self.img_gt = self.label[self.data_index: self.data_index + 1, ...]
            # self.img = self.data[self.data_index, ...]
            # self.img_gt = self.label[self.data_index, ...]
            # self.psnr, pred_label = prob_cal(self.reward_net, self.img, self.img_gt)
            self.psnr, pred_label = self.eval_engine.prob_cal(self.img, self.img_gt, k=self.config.reward_mul)
            # if self.psnr > 1:  # ignore too smooth samples and rule out 'inf'
            #     self.data_index += 1
            # else:
            break

        # update training file
        if self.data_index >= self.data_len:
            if self.train_max > 1:
                self.train_cur += 1
                if self.train_cur >= self.train_max:
                    self.train_cur = 0

                # load new file
                print('loading file No.%d' % (self.train_cur + 1))
                f = h5py.File(self.train_list[self.train_cur], 'r')
                self.data = f['data'].value
                self.label = f['label'].value
                if hasattr(self.config, 'train_idx') and hasattr(self.config, 'valid_idx'):
                    self.data = self.data[self.config.valid_idx,]
                    self.label = self.label[self.config.valid_idx]
                self.data_len = len(self.data)
                f.close()

            # start from beginning
            self.data_index = 0
            while True:
                self.img = self.data[self.data_index: self.data_index + 1, ...]
                self.img_gt = self.label[self.data_index: self.data_index + 1]
                self.psnr, pred_label = self.eval_engine.prob_cal(self.img, self.img_gt, k=self.config.reward_mul)
                # self.psnr, pred_label = prob_cal(self.reward_net, self.img, self.img_gt)
                if pred_label != int(self.img_gt):  # ignore samples that cannot be correctly predicted by classifier
                    self.data_index += 1
                else:
                    break

        self.reward = 0
        self.count = 0
        self.psnr_init = self.psnr
        self.data_index += 1
        return self.img, self.reward, 0, self.terminal

    def parse_action(self, num, mags, mags_cum):
        action = np.searchsorted(mags_cum, num, side='right')
        if action == 0:
            base = 0
        else:
            base = mags_cum[action - 1]
        mag = mags[action][num - base] if action < len(mags_cum) else 0
        mag /= 10
        return action, mag

    def act(self, action):
        self.psnr_pre = self.psnr
        if action == self.action_size - 1:  # stop
            self.terminal = True
        else:
            # aug_method = action // self.config.n_mag
            # aug_magnitude = (action % self.config.n_mag - 0.5) / 5 + 0.5
            aug_method, aug_magnitude = self.parse_action(action, self.config.mags, self.config.mags_cum)
            im_out = apply_augment(self.img, aug_method, aug_magnitude)
            self.img = im_out
        # self.psnr, pred_label = prob_cal(self.reward_net, self.img, self.img_gt)
        self.psnr, pred_label = self.eval_engine.prob_cal(self.img, self.img_gt, k=self.config.reward_mul)

        # max step
        if self.count >= self.stop_step - 1:
            self.terminal = True

        # calculate reward
        self.reward = self.reward_function(self.psnr, self.psnr_pre)
        self.count += 1

        # stop if too bad, penalize the action
        if pred_label != int(self.img_gt):
            self.terminal = True
            self.reward = -self.reward

        return self.img, self.reward, self.terminal


    def act_test(self, action, step = 0):
        reward_all = np.zeros(action.shape)
        psnr_all = np.zeros(action.shape)
        acc_all = np.zeros(action.shape)
        if step == 0:
            self.test_imgs = self.data_test.copy()
            self.test_temp_imgs = self.data_test.copy()
            self.test_pre_imgs = self.data_test.copy()
            self.test_steps = np.zeros(len(action), dtype=int)
        for k in range(len(action)):
            img_in = self.data_test[k:k+1,...].copy() if step == 0 else self.test_imgs[k:k+1,...].copy()
            img_label = self.label_test[k:k+1,...].copy()
            self.test_temp_imgs[k:k+1,...] = img_in.copy()
            # psnr_pre, pred_label = prob_cal(self.reward_net, img_in, img_label)
            psnr_pre, pred_label = self.eval_engine.prob_cal(img_in, img_label, k=self.config.reward_mul)
            pre_pred_label = pred_label
            # if action[k] == self.action_size - 1 or self.test_steps[k] == self.stop_step: # stop action or already stop
            if int(pred_label) != int(img_label): # if has wrong prediction, make the action stop
                action[k] = self.action_size - 1
            if action[k] == self.action_size - 1 or self.test_steps[k] == self.stop_step: # stop action or already stop
                img_out = img_in.copy()
                self.test_steps[k] = self.stop_step # terminal flag
            else:
                # aug_method = action[k] // self.config.n_mag
                # aug_magnitude = (action[k] % self.config.n_mag - 0.5) / 5 + 0.5
                aug_method, aug_magnitude = self.parse_action(action[k], self.config.mags, self.config.mags_cum)
                img_out = apply_augment(img_in, aug_method, aug_magnitude)
                self.test_steps[k] += 1
            self.test_pre_imgs[k:k+1,...] = self.test_temp_imgs[k:k+1,...].copy()
            self.test_imgs[k:k+1,...] = img_out.copy()  # keep intermediate results
            # psnr, pred_label = prob_cal(self.reward_net, img_out, img_label)
            psnr, pred_label = self.eval_engine.prob_cal(img_out, img_label, k=self.config.reward_mul)

            if int(pred_label) != int(img_label): # if has wrong prediction, cancel the action, make reward=0
                action[k] = self.action_size - 1
                self.test_steps[k] = self.stop_step  # terminal flag
                psnr = psnr_pre
                pred_label = pre_pred_label
                self.test_imgs[k:k + 1, ...] = img_in.copy()

            reward = self.reward_function(psnr, psnr_pre=psnr_pre)
            psnr_all[k] = psnr
            reward_all[k] = reward
            acc_all[k] = 1 if pred_label == img_label else 0

        if self.is_train:
            return reward_all.mean(), psnr_all.mean(), acc_all.mean(), self.base_psnr, self.base_acc
        else:
            return reward_all, psnr_all, acc_all, self.base_psnr, self.base_acc


    def update_test_data(self):
        self.test_cur = self.test_cur + len(self.data_test)
        test_end = min(self.test_total, self.test_cur + self.test_batch)
        if self.test_cur >= test_end:
            return False #failed
        else:
            self.data_test = self.data_all[self.test_cur: test_end, ...]
            self.label_test = self.label_all[self.test_cur: test_end, ...]

            # update base psnr
            self.base_psnr = 0.
            self.base_acc = 0.
            # for k in range(len(self.data_test)):
            #     # psnr, pred_label = prob_cal(self.reward_net, self.data_test[k, ...], self.label_test[k, ...])
            #     psnr, pred_label = self.eval_engine.prob_cal(self.data_test[k, ...], self.label_test[k, ...])
            #     self.base_psnr += psnr
            #     self.base_acc += 1 if pred_label == int(self.label_test[k, ...]) else 0
            for k in range(0, len(self.data_test), self.config.test_batch_size):
                psnr, pred_label= self.eval_engine.prob_cal_batch(self.data_test[k:k+self.config.test_batch_size, ...],
                                                            self.label_test[k:k+self.config.test_batch_size, ...],
                                                                  k=self.config.reward_mul)
                self.base_psnr += np.sum(psnr)
                self.base_acc += np.sum(pred_label == self.label_test[k:k+self.config.test_batch_size, ...])
            self.base_psnr /= len(self.data_test)
            self.base_acc /= len(self.data_test)
            return True #successful


    # def act_test_mine(self, my_img_cur, action):
    #     if action == self.action_size - 1:
    #         return my_img_cur.copy()
    #     else:
    #         if my_img_cur.ndim == 4:
    #             feed_img_cur = my_img_cur
    #         else:
    #             feed_img_cur = my_img_cur.reshape((1,) + my_img_cur.shape)
    #         my_img_next = self.sessions[action].run(self.outputs[action], feed_dict={self.inputs[action]: feed_img_cur})
    #         return my_img_next[0, ...]


    def update_test_mine(self):
        """
        :return: (image, image name) or (None, None)
        """
        if self.my_img_idx >= len(self.my_img_list):
            return None, None
        else:
            img_name = self.my_img_list[self.my_img_idx]
            base_name, _ = os.path.splitext(img_name)
            my_img = cv2.imread(self.my_img_dir + img_name)
            my_img = my_img[:,:,::-1] / 255.
            self.my_img_idx += 1
            return my_img, base_name


    def get_test_imgs(self):
        return self.test_imgs.copy()


    def get_test_steps(self):
        return self.test_steps.copy()


    def get_data_test(self):
        return self.data_test.copy()


    def get_test_info(self):
        return self.test_cur, len(self.data_test) # current image number & batch size
