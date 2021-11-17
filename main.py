import os
import sys
import numpy as np
from time import time
from datetime import timezone, timedelta,datetime
import torch
GPU = torch.cuda.is_available()

parent = os.path.dirname(sys.path[0])#os.getcwd())
sys.path.append(parent)
from util import SimpleLogger,visualize_prediction_compare

from tensorboard_logger import configure, log_value
import argparse
import os
import pickle
import sys
import traceback
import shutil

parser = argparse.ArgumentParser(description='Models for Continuous Stirred Tank dataset')

parser.add_argument("--save", type=str, default='results', help="experiment logging folder")

paras = parser.parse_args()

hard_reset = paras.reset

# if paras.save already exists and contains log.txt:
# reset if not finished, or if hard_reset
paras.save = os.path.join('results', paras.save) #路径拼接，改变paras.save为'results/tmp'
if paras.test:
    model_test_path = os.path.join(paras.save, 'best_dev_model.pt')
    paras.save = os.path.join(paras.save, 'test')
    if not os.path.exists(paras.save):
        os.mkdir(paras.save)


log_file = os.path.join(paras.save, 'log.txt')
if os.path.isfile(log_file) and not paras.test:  #判断是否为文件
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        completed = 'Finished' in content
        if 'tmp' not in paras.save and completed and not hard_reset:
            print('Exit; already completed and no hard reset asked.')
            sys.exit()  # do not overwrite folder with current experiment
        else:  # reset folder
            shutil.rmtree(paras.save, ignore_errors=True) #递归地删除文件夹


logging = SimpleLogger(log_file) #log to file
configure(paras.save) #tensorboard logging
logging('Args: {}'.format(paras))

GPU = torch.cuda.is_available()
logging('Using GPU?', GPU)



logging('Finished: best dev error')