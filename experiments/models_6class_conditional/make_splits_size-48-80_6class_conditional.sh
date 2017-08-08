DATA_PATH='../../data'
IMAGE_SIZE='48-80'
DATASET_SOURCE="LIN_Normalized_WT_size-${IMAGE_SIZE}"

SPLITS_TEST="${DATASET_SOURCE}_6class_test_splits"

TARGET_PATH_TEST="${DATA_PATH}/${DATASET_SOURCE}_6class_conditional_test_splits"
mkdir -p ${TARGET_PATH}

CLASSES='Alp14 Arp3 Cki2 Mkh1 Sid2 Tea1'
NUM_SPLITS=10
ROLES='train test'

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
		done
	done
done
