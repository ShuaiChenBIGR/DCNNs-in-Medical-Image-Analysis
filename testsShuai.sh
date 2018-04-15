#!/bin/bash

NAME_QUEUE="day"
MEMORY="50G"
JOB_NAME="HELLO_SHUAI_post_processing_CRF" 

LOGS_DIR="/scratch/schen/post_processing_CRF/build/applicationAndExamples"
mkdir -p $LOGS_DIR
FILE_LOGS=${LOGS_DIR}/${JOB_NAME}.log

EXECUTABLE="/scratch/schen/post_processing_CRF/build/applicationAndExamples/dense3DCrfInferenceOnNiis"
CONFIG_FILE="/scratch/schen/post_processing_CRF/applicationAndExamples/example/configFileDenseCrf3d.txt"


CALL="${EXECUTABLE} -c ${CONFIG_FILE}"

echo "$CALL" | qsub -q ${NAME_QUEUE} -l h_vmem=${MEMORY} -j y -N ${JOB_NAME} -o ${FILE_LOGS}
