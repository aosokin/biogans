ROOT_PATH='../..'
MODEL_NAME='size-48-80_6class_wgangp-star-shaped-adam'
NET_PATH="${ROOT_PATH}/models/${MODEL_NAME}/netG_iter_50000.pth"

RESULT_PATH="./reconstruction/${MODEL_NAME}"
mkdir -p ${RESULT_PATH}

python ${ROOT_PATH}/code/main.py --dataroot ${ROOT_PATH}/data/LIN_Normalized_WT_size-48-80_6class_conditional_train --image_height 48 --image_width 80 --cuda --num_iter 50000 --experiment ${RESULT_PATH} --dataset_type conditional-cached --model_type DCGAN-star-shaped --mode reconstruction --netG ${NET_PATH} --test_data ${ROOT_PATH}/data/LIN_Normalized_WT_size-48-80_6class_conditional_test | tee ${RESULT_PATH}/reconstruction_stdout.txt


RESULT_PATH="./reconstruction-separable/${MODEL_NAME}"
mkdir -p ${RESULT_PATH}

python ${ROOT_PATH}/code/main.py --dataroot ${ROOT_PATH}/data/LIN_Normalized_WT_size-48-80_6class_conditional_train --image_height 48 --image_width 80 --cuda --num_iter 50000 --experiment ${RESULT_PATH} --dataset_type conditional-cached --model_type DCGAN-star-shaped --mode reconstruction-separable --netG ${NET_PATH} --test_data ${ROOT_PATH}/data/LIN_Normalized_WT_size-48-80_6class_conditional_test | tee ${RESULT_PATH}/reconstruction_stdout.txt
