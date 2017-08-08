import os

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(CUR_PATH, '..', '..')

EVAL_OUTPUT_PATH = '{0}/logs_eval_8class_real_vs_real'.format(CUR_PATH)

DATA_PATH = '{0}/data/LIN_Normalized_WT_size-48-80_8class_real_vs_real_splits'.format(ROOT_PATH)
DATA_PATH_A = '{0}/train'.format(DATA_PATH)
DATA_PATH_B = '{0}/test'.format(DATA_PATH)

NUM_DITER = 5000
JOB_PATH = '{0}/jobs_eval_8class_real_vs_real'.format(CUR_PATH)

os.system('mkdir -p {0}'.format(JOB_PATH))
os.system('mkdir -p {0}'.format(EVAL_OUTPUT_PATH))

NUM_SPLITS = 10
RAND_SEED = 0
CLASSES = ['Alp14', 'Arp3', 'Cki2', 'Mkh1', 'Sid2', 'Tea1', 'Fim1', 'Tea4']

EVAL_METHODS = ['WGAN-GP']
EVAL_METHOD_PARAMS = ['--model_type DCGAN --optimizer adam --lrD 0.0001 --lrG 0.0001 --beta1 0.0 --beta2 0.9']

for SPLIT_ID in range(NUM_SPLITS):
    for CLASS_A in CLASSES:
        for CLASS_B in CLASSES:
            for i_eval, EVAL_METHOD in enumerate(EVAL_METHODS):

                JOB_NAME = 'seed{0}_split{1}_eval{2}_iter{3}_classes{4}{5}'.format(
                    RAND_SEED, SPLIT_ID, EVAL_METHOD, NUM_DITER, CLASS_A, CLASS_B)

                JOB_FILE = JOB_PATH + '/' + JOB_NAME + '.sh'

                DATA_TRAIN_A = '{0}/split{1}/train/{2}'.format(DATA_PATH_A, SPLIT_ID, CLASS_A)
                DATA_TRAIN_B = '{0}/split{1}/train/{2}'.format(DATA_PATH_B, SPLIT_ID, CLASS_B)
                DATA_TEST_A = '{0}/split{1}/test/{2}'.format(DATA_PATH_A, SPLIT_ID, CLASS_A)
                DATA_TEST_B = '{0}/split{1}/test/{2}'.format(DATA_PATH_B, SPLIT_ID, CLASS_B)

                job_template = """
export ROOT_PATH={ROOT_PATH}
cd ${{ROOT_PATH}}

export EXP_PATH={EVAL_OUTPUT_PATH}
mkdir -p ${{EXP_PATH}}

python ${{ROOT_PATH}}/code/main.py --random_seed {RAND_SEED} --image_height 48 --image_width 80 --dataset_type folder-cached --num_workers 0 --cuda  --mode eval-real-vs-real --GAN_algorithm {EVAL_METHOD} --num_iter {NUM_DITER} {EVAL_PARAMS} --experiment {EVAL_OUTPUT_PATH} --dataroot {DATA_TRAIN_A} --test_data {DATA_TRAIN_B} --extra_test_sets {DATA_TEST_A} {DATA_TEST_B} > {EVAL_OUTPUT_PATH}/{JOB_NAME}.txt
"""

                job = job_template.format(ROOT_PATH=ROOT_PATH, EVAL_OUTPUT_PATH=EVAL_OUTPUT_PATH,
                                          JOB_NAME=JOB_NAME, EVAL_METHOD=EVAL_METHOD, RAND_SEED=RAND_SEED, NUM_DITER=NUM_DITER,
                                          EVAL_PARAMS=EVAL_METHOD_PARAMS[i_eval],
                                          DATA_TRAIN_A=DATA_TRAIN_A, DATA_TRAIN_B=DATA_TRAIN_B,
                                          DATA_TEST_A=DATA_TEST_A, DATA_TEST_B=DATA_TEST_B
                                          )

                with open(JOB_FILE, "w") as f:
                    print("{}".format(job), file=f)
