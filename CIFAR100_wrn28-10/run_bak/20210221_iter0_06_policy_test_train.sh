python main.py --is_train=False --dataset=cifar10 --play_model=models/20210221_iter0_06/ --stop_step=10 --save_h5=data/augmentation/20210221_iter0_06_train --reverse_reward=True --test_dir=data/cifar10/train/ --class_model=autoaug2/training/20210221_iter0_06/model.ckpt-199  --reward_mul=10 | tee models/20210221_iter0_06/test_output_train.log