python main.py --is_train=False --dataset=cifar100 --play_model=models/20210113_iter2_06/ --stop_step=10 --save_h5=data/augmentation/20210113_iter2_06_test --reverse_reward=True --test_dir=data/cifar100/test/ --class_model=autoaug2/training/20210113_iter2_06/model.ckpt-199  --reward_mul=10 | tee models/20210113_iter2_06/test_output_test.log
