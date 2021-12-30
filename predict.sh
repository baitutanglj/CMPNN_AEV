#!/bin/bash
source /mnt/home/linjie/anaconda3/bin/activate chemprop
python predict.py --data_path /mnt/home/linjie/projects/CMPNN-master/data/small_tox21.csv \
                  --checkpoint_dir /mnt/home/linjie/projects/CMPNN-master/ckpt_debug