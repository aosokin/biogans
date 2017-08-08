ROOT_PATH='../..'
RESULT_PATH="${ROOT_PATH}/models/size-48-80_6class_wgangp-multichannel-sep-adam"

mkdir -p ${RESULT_PATH}

python ${ROOT_PATH}/code/main.py --dataroot ${ROOT_PATH}/data/LIN_Normalized_WT_size-48-80_6class_train --image_height 48 --image_width 80 --cuda --num_iter 50000 --experiment ${RESULT_PATH} --dataset_type fake-multichannel --model_type DCGAN-sep-multichannel --red_nn_file ${ROOT_PATH}/data/red_neighbors_LIN_Normalized_WT_size-48-80_all_to_train.pth --nz 350 --ngf 224 --ndf 224 --red_portion 0.143 --GAN_algorithm WGAN-GP --num_disc_iters 5 --optimizer adam --lrD 0.0001 --lrG 0.0001 --beta1 0.0 --beta2 0.9 | tee ${RESULT_PATH}/train_stdout.txt
