import os

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(CUR_PATH, '..', '..')

EVAL_OUTPUT_PATH = '{0}/logs_eval_6class_conditional_fake_vs_real'.format(CUR_PATH)
DATA_PATH = '{0}/data/LIN_Normalized_WT_size-48-80_6class_conditional_test_splits'.format(ROOT_PATH)
DATA_ROOT = "{0}/data/LIN_Normalized_WT_size-48-80_6class_conditional_test_splits".format(ROOT_PATH)
NET_PATH = '{0}/models'.format(ROOT_PATH)

NUM_DITER = 5000
FIXED_NOISE_FILE_100 = '{0}/fixed_noise_batch64_dim100.pth'.format(CUR_PATH)
FIXED_NOISE_FILE_350 = '{0}/fixed_noise_batch64_dim350.pth'.format(CUR_PATH)
JOB_PATH = '{0}/jobs_eval_6class_conditional_fake_vs_real'.format(CUR_PATH)

os.system('mkdir -p {0}'.format(JOB_PATH))
os.system('mkdir -p {0}'.format(EVAL_OUTPUT_PATH))

NUM_SPLITS = 10
RAND_SEED = 0

MODEL_NAMES = ['size-48-80_6class_wgangp-star-shaped-adam',
               'size-48-80_6class_wgangp-independent-adam',
               'size-48-80_6class_wgangp-independent-sep-adam',
               'size-48-80_6class_wgangp-multichannel-adam',
               'size-48-80_6class_wgangp-multichannel-sep-adam']
MODEL_PARAMS = ['--dataset_type conditional-cached --model_type DCGAN-star-shaped --fixed_noise_file ' + FIXED_NOISE_FILE_100,
                '--dataset_type conditional-cached --model_type DCGAN-independent --fixed_noise_file ' + FIXED_NOISE_FILE_100,
                '--dataset_type conditional-cached --model_type DCGAN-sep-independent --fixed_noise_file ' + FIXED_NOISE_FILE_100,
                '--dataset_type conditional-cached --model_type DCGAN-multichannel --nz 350 --ngf 224 --ndf 64 --fixed_noise_file ' + FIXED_NOISE_FILE_350,
                '--dataset_type conditional-cached --model_type DCGAN-sep-multichannel --nz 350 --ngf 224 --ndf 64 --red_portion 0.143 --fixed_noise_file ' + FIXED_NOISE_FILE_350]

MODEL_ITER_OF_INTEREST = [50000]
EVAL_METHODS = ['GAN', 'WGAN', 'WGAN-GP']
EVAL_METHOD_PARAMS = ['--optimizer adam --lrD 0.0002 --lrG 0.0002 --beta1 0.5 --beta2 0.999',
                      '--optimizer rmsprop --lrD 0.00005 --lrG 0.00005',
                      '--optimizer adam --lrD 0.0001 --lrG 0.0001 --beta1 0.0 --beta2 0.9']

for SPLIT_ID in range(NUM_SPLITS):
    for i_model, MODEL_NAME in enumerate(MODEL_NAMES):
        for MODEL_ITER in MODEL_ITER_OF_INTEREST:
            for i_eval, EVAL_METHOD in enumerate(EVAL_METHODS):

                NETWORK_NAME = '{0}_trainIter{1}'.format(MODEL_NAME, MODEL_ITER)

                JOB_NAME = 'seed{0}_split{1}_{2}_eval{3}_iter{4}'.format(
                    RAND_SEED, SPLIT_ID, NETWORK_NAME, EVAL_METHOD, NUM_DITER)

                JOB_FILE = JOB_PATH + '/' + JOB_NAME + '.sh'

                job_template = """
export ROOT_PATH={ROOT_PATH}
cd ${{ROOT_PATH}}

export EXP_PATH={EVAL_OUTPUT_PATH}
mkdir -p ${{EXP_PATH}}

python ${{ROOT_PATH}}/code/main.py --random_seed {RAND_SEED} --dataroot {DATA_ROOT}/split{SPLIT_ID}/train --image_height 48 --image_width 80 --num_workers 0 --cuda {MODEL_PARAMS} --mode eval-gen-vs-real --netG {NET_PATH}/{MODEL_NAME}/netG_iter_{MODEL_ITER}.pth --test_data {DATA_ROOT}/split{SPLIT_ID}/test --GAN_algorithm {EVAL_METHOD} --num_iter {NUM_DITER} {EVAL_PARAMS} --experiment {EVAL_OUTPUT_PATH} --prefix_fake_samples {SAMPLE_FILES_PREFIX} > {EVAL_OUTPUT_PATH}/{JOB_NAME}.txt
"""

                job = job_template.format(ROOT_PATH=ROOT_PATH, EVAL_OUTPUT_PATH=EVAL_OUTPUT_PATH, NET_PATH=NET_PATH,
                                          JOB_NAME=JOB_NAME, DATA_ROOT=DATA_ROOT, SPLIT_ID=SPLIT_ID,
                                          MODEL_NAME=MODEL_NAME, MODEL_PARAMS=MODEL_PARAMS[i_model], MODEL_ITER=MODEL_ITER,
                                          EVAL_METHOD=EVAL_METHOD, RAND_SEED=RAND_SEED, NUM_DITER=NUM_DITER,
                                          EVAL_PARAMS=EVAL_METHOD_PARAMS[i_eval],
                                          SAMPLE_FILES_PREFIX=NETWORK_NAME)

                with open(JOB_FILE, "w") as f:
                    print("{}".format(job), file=f)
