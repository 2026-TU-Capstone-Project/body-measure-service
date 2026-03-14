#!/bin/bash
uname -a
date

RESTORE_FROM='./weights/NIA2_epoch_139.pth'

python evaluate.py --restore-from ${RESTORE_FROM}
