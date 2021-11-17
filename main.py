import os
import sys
import numpy as np
from time import time
from datetime import timezone, timedelta,datetime
import torch
from  torch import nn
GPU = torch.cuda.is_available()

parent = os.path.dirname(sys.path[0])#os.getcwd())
sys.path.append(parent)
from util import SimpleLogger,visualize_prediction_compare
from deal_dataset import *
from lstm import LSTM
from tensorboard_logger import configure, log_value
import argparse
import os
import pickle
import sys
import traceback
import shutil

parser = argparse.ArgumentParser(description='Models for Continuous Stirred Tank dataset')
parser.add_argument("--test", action="store_true", help="Testing model in para.save")
parser.add_argument("--seed", type=int, default=None, help="random seed")
parser.add_argument("--save", type=str, default='results', help="experiment logging folder")
parser.add_argument("--reset", action="store_true", help="reset even if same experiment already finished")
parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
parser.add_argument("--eval_epochs", type=int, default=10, help="validation every so many epochs")
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

#设置gpu的随机数种子
if paras.seed is None:
    paras.seed = np.random.randint(1000000)
torch.manual_seed(paras.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(paras.seed)
np.random.seed(paras.seed)

logging('Random seed', paras.seed)

dm = Data_Manager()
dm.load_train_data() #加载训练集
dm.load_test_data() #加载测试集
#将不存在的数据转成num
np.nan_to_num(dm.X_, copy=False)
np.nan_to_num(dm.X0, copy=False)
np.nan_to_num(dm.Y0, copy=False)
np.nan_to_num(dm.S, copy=False)
np.nan_to_num(dm.W, copy=False)
np.nan_to_num(dm.test_X_, copy=False)
np.nan_to_num(dm.test_X0, copy=False)
np.nan_to_num(dm.test_W, copy=False)
times = 1
dm.generate_indexes(times) #验证集索引
dm.generate_dev() #生成验证集和验证集结果


if paras.mymodel == 'lstm':
    model = LSTM()



model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)  #优化

# try:
#     for epoch in range(1, paras.epochs + 1):
#
#
#         print("训练")
#         if epoch % paras.eval_epochs == 0:
#             with torch.no_grad(): #验证  停止梯度计算
#                 model.eval()      #测试
#
# except:
#     var = traceback.format_exc()
#     logging(var)
#
# logging('Finished: best dev error')