import os

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(CUR_PATH, '..', '..')

EVAL_OUTPUT_PATH = '{0}/logs_eval_6class-together_real_vs_real'.format(CUR_PATH)

DATA_PATH_A = '{0}/data/LIN_Normalized_WT_size-48-80_6class_test_splits'.format(ROOT_PATH)
DATA_PATH_B = '{0}/data/LIN_Normalized_WT_size-48-80_6class_train_splits'.format(ROOT_PATH)

NUM_DITER = 5000
JOB_PATH = '{0}/jobs_eval_6class-together_real_vs_real'.format(CUR_PATH)

os.system('mkdir -p {0}'.format(JOB_PATH))
os.system('mkdir -p {0}'.format(EVAL_OUTPUT_PATH))

NUM_SPLITS = 10
RAND_SEED = 0

EVAL_METHODS = ['GAN', 'WGAN', 'WGAN-GP']
EVAL_METHOD_PARAMS = ['--model_type DCGAN --optimizer adam --lrD 0.0002 --lrG 0.0002 --beta1 0.5 --beta2 0.999',
                      '--model_type DCGAN --optimizer rmsprop --lrD 0.00005 --lrG 0.00005',
                      '--model_type DCGAN --optimizer adam --lrD 0.0001 --lrG 0.0001 --beta1 0.0 --beta2 0.9']

for SPLIT_ID in range(NUM_SPLITS):
    for i_eval, EVAL_METHOD in enumerate(EVAL_METHODS):

        JOB_NAME = 'seed{0}_split{1}_eval{2}_iter{3}'.format(
            RAND_SEED, SPLIT_ID, EVAL_METHOD, NUM_DITER)

        JOB_FILE = JOB_PATH + '/' + JOB_NAME + '.sh'

        DATA_TRAIN_A = '{0}/split{1}/train'.format(DATA_PATH_A, SPLIT_ID)
        DATA_TRAIN_B = '{0}/split{1}/train'.format(DATA_PATH_B, SPLIT_ID)
        DATA_TEST_A = '{0}/split{1}/test'.format(DATA_PATH_A, SPLIT_ID)
        DATA_TEST_B = '{0}/split{1}/test'.format(DATA_PATH_B, SPLIT_ID)

        job_template = """
export ROOT_PATH={ROOT_PATH}
cd ${{ROOT_PATH}}

export EXP_PATH={EVAL_OUTPUT_PATH}
mkdir -p ${{EXP_PATH}}

python ${{ROOT_PATH}}/code/main.py --random_seed {RAND_SEED} --image_height 48 --image_width 80 --dataset_type folder-cached --num_workers 0 --cuda  --mode eval-real-vs-real --GAN_algorithm {EVAL_METHOD} --num_iter {NUM_DITER} {EVAL_PARAMS} --experiment {EVAL_OUTPUT_PATH} --dataroot {DATA_TRAIN_A} --test_data {DATA_TRAIN_B} --extra_test_sets {DATA_TEST_A} {DATA_TEST_B} > {EVAL_OUTPUT_PATH}/{JOB_NAME}.txt
"""

        job = job_template.format(ROOT_PATH=ROOT_PATH, EVAL_OUTPUT_PATH=EVAL_OUTPUT_PATH, JOB_NAME=JOB_NAME,
                                  EVAL_METHOD=EVAL_METHOD, RAND_SEED=RAND_SEED, NUM_DITER=NUM_DITER,
                                  EVAL_PARAMS=EVAL_METHOD_PARAMS[i_eval],
                                  DATA_TRAIN_A=DATA_TRAIN_A, DATA_TRAIN_B=DATA_TRAIN_B,
                                  DATA_TEST_A=DATA_TEST_A, DATA_TEST_B=DATA_TEST_B
                                  )

        with open(JOB_FILE, "w") as f:
            print("{}".format(job), file=f)
