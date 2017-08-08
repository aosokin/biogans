ROOT_PATH='../..'
RESULT_PATH="${ROOT_PATH}/models/size-48-80_6class_gan-independent-adam"

mkdir -p ${RESULT_PATH}

python ${ROOT_PATH}/code/main.py --dataroot ${ROOT_PATH}/data/LIN_Normalized_WT_size-48-80_6class_conditional_train/ --image_height 48 --image_width 80 --cuda --num_iter 50000 --experiment ${RESULT_PATH} --dataset_type conditional-cached --model_type DCGAN-independent --GAN_algorithm GAN --optimizer adam --lrD 0.0002 --lrG 0.0002 --beta1 0.5 --beta2 0.999 | tee ${RESULT_PATH}/train_stdout.txt
