set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python example\train_hand21_112.py --gpus 1 --lr 0.01 --image_set hand_landmarks21_cut --end_epoch 8000 --lr_epoch 2500,5000,8000 --batch_size 100 --thread_num 10 --frequent 10 
pause 