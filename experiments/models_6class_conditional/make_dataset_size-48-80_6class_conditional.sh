DATA_PATH='../../data'
IMAGE_SIZE='48-80'
DATASET_SOURCE='LIN_Normalized_WT_size-'${IMAGE_SIZE}
DATASETS_PATH=${DATASET_SOURCE}'_6class_conditional'

CLASSES='Alp14 Arp3 Cki2 Mkh1 Sid2 Tea1'
SPLITS='train test'

for SPLIT in ${SPLITS};
do
	for CLASS in ${CLASSES};
	do
		SOURCE_PATH=${DATA_PATH}/${DATASET_SOURCE}_${SPLIT}
		TARGET_PATH=${DATA_PATH}/${DATASETS_PATH}_${SPLIT}/${CLASS}
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
