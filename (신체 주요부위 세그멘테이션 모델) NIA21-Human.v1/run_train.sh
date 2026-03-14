#!/bin/bash
uname -a
date

DATA_PATH='./temp/trainval'
GPU_IDS=0,1,2,3
DATASET='trainval'
EPOCHS=250
SNAPSHOT_DIR='./temp/snapshot'

if [[ ! -e ${SNAPSHOT_DIR} ]]; then
    mkdir -p  ${SNAPSHOT_DIR}
fi

python train.py --data-dir ${DATA_PATH} \
       --gpu ${GPU_IDS}\
       --snapshot-dir ${SNAPSHOT_DIR}\
       --dataset ${DATASET}\
       --epochs ${EPOCHS}
