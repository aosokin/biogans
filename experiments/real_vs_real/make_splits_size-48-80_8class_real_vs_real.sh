DATA_PATH='../../data'
IMAGE_SIZE='48-80'
DATASET_SOURCE='LIN_Normalized_WT_size-'${IMAGE_SIZE}

SPLITS_TEST=${DATASET_SOURCE}_8class_test_splits
SPLITS_TRAIN=${DATASET_SOURCE}_8class_train_splits

TARGET_PATH=${DATA_PATH}/${DATASET_SOURCE}_8class_real_vs_real_splits
mkdir -p ${TARGET_PATH}
TARGET_PATH_TEST=${TARGET_PATH}/test
TARGET_PATH_TRAIN=${TARGET_PATH}/train
mkdir ${TARGET_PATH_TEST} ${TARGET_PATH_TRAIN}

CLASSES='Alp14 Arp3 Cki2 Mkh1 Sid2 Tea1 Fim1 Tea4'
NUM_SPLITS=10
ROLES='test train'
for SPLIT in `seq 0 $((NUM_SPLITS-1))`;
do
	for CLASS in ${CLASSES};
	do
		for ROLE in $ROLES;
		do
			# test splits
			SOURCE_PATH=${DATA_PATH}/${SPLITS_TEST}/split${SPLIT}/${ROLE}
			TARGET_PATH=${TARGET_PATH_TEST}/split${SPLIT}/${ROLE}/${CLASS}
			mkdir -p ${TARGET_PATH}
			if [ -d ${TARGET_PATH}/${CLASS} ]; then
				echo 'Removing old '${TARGET_PATH}/${CLASS}
				if [ -L ${TARGET_PATH}/${CLASS} ]; then
					rm ${TARGET_PATH}/${CLASS}
				else
					rmdir ${TARGET_PATH}/${CLASS}
				fi
				rm -rf ${TARGET_PATH}/${CLASS}
			fi
			echo 'Creating '${TARGET_PATH}/${CLASS}			
			ln -rs ${SOURCE_PATH}/${CLASS} ${TARGET_PATH}/${CLASS}

			# train splits
			SOURCE_PATH=${DATA_PATH}/${SPLITS_TRAIN}/split${SPLIT}/${ROLE}
			TARGET_PATH=${TARGET_PATH_TRAIN}/split${SPLIT}/${ROLE}/${CLASS}
			mkdir -p ${TARGET_PATH}

			if [ -d ${TARGET_PATH}/${CLASS} ]; then
				echo 'Removing old '${TARGET_PATH}/${CLASS}
				if [ -L ${TARGET_PATH}/${CLASS} ]; then
					rm ${TARGET_PATH}/${CLASS}
				else
					rmdir ${TARGET_PATH}/${CLASS}
				fi
				rm -rf ${TARGET_PATH}/${CLASS}
			fi
			echo 'Creating '${TARGET_PATH}/${CLASS}			
			ln -rs ${SOURCE_PATH}/${CLASS} ${TARGET_PATH}/${CLASS}
		done
	done
done
