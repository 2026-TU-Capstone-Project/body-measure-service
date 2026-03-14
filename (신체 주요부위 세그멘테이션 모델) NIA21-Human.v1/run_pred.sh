#!/bin/bash
uname -a
date

DATA_PATH='input_data'
RESTORE_FROM='./weights/NIA2_epoch_139.pth'
OUT_PATH='./output/pred'

python predict.py --data-dir ${DATA_PATH} \
       --restore-from ${RESTORE_FROM}\
       --output-path ${OUT_PATH}
