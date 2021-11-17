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
from util import SimpleLogger,visualize_prediction_compare,Compare_dev_result,cal_R
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
parser.add_argument("--mymodel", type=str, default='merge', choices=['lstm'])
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
#将不存在的数据转成0，现在训练集的数据有些地方不存在，强制转成0了都
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
dm.generate_dev() #生成验证集和验证集结果以及验证集索引，写入csv文件。

"++++++++++++++++++++++++++++++++++围起来的地儿是为了说明数据的访问方法，以及可视化的使用，可以注释掉+++++++++++++++++++++++++++++++++++++++++++"

print("-----------------------根据索引遍历训练集----------------------------")
# 训练集遍历功率温度，风向风速
#用法,
for f in range(field_len):
    # 可以看到，这就是无放回的抽样，每个风场训练集共有17507个时段，每次取出春夏秋冬各20段，也就是共80段，这80段是25个风场的数据，每次索引共能取出80*25共2000个时段，用以训练。。。一共能取17507/80次的索引,看来数据也不少
    #我们要做的就是综合输入 dm.X_[f, indexes]，dm.X0[f, indexes]，dm.W[f, indexes].选择合适的模型，来预测y（未来10分钟的风速，风向）, 并与dm.Y0[f, indexes]求loss,优化
    indexes = dm.get_indexes(f)  #取训练集索引，并随机化,
    indexes1 = dm.get_indexes(f)
    indexes2 = dm.get_indexes(f)
    print(indexes)
    print(field[f], dm.X_[f, indexes].shape) #功率 温度  风场f (80, 25, 120, 2)
    print(field[f], dm.X0[f, indexes].shape) #风速 风向 风场f (80, 25, 120, 2)
    print(field[f], dm.W[f, indexes].shape)  # 对应的气象数据（风速风向，未归一化）  风场f (80, 14, 2)，共13个小时的数据，也就是14个值，前12个值是过去的，后两个是未来的 ,所有风机都是用的这个

    print(field[f], dm.Y0[f, indexes].shape) #输出的风向风速   风场1 (80, 25, 20, 2)
print("----------------------遍历测试集-----------------------------")

#测试集按照这种方式遍历，最后将测试集中的数据，输入到训练好的模型中，得出结果，写入到文件中，就是我们要提交的答案，直接看print输出应该好理解
for f in range(field_len):   #风场
    for p in range(period_len):  #时段
        for m in range(machine_len):  #机器
            print(field[f], period[p], machine[f][m], dm.test_X_[f, p, m].shape) #（功率 温度）  print 风场f 时段？  风机号  (120, 2)
            print(field[f], period[p], machine[f][m], dm.test_X0[f, p, m].shape)#（风速 风向）   print 风场f 时段？  风机号  (120, 2)
            print(field[f], period[p], dm.test_W[f, p].shape)  # 对应的气象数据（风速风向，未归一化）  print 风场f 时段？   (14, 2),一个风场所有风机都是用的这个

print("-----------------------遍历验证集----------------------------")
#验证集遍历，只有这80个时段,用于我们评估
val_indexes_df = read_csv('./data/val_indexes.csv')
val_indexes = {field[f]: {0: val_indexes_df[field[f]].values} for f in range(field_len)}
for f in range(field_len):
    indexes = dm.get_indexes(f)
    print(field[f], dm.X_[f, indexes].shape)  # 功率温度  风场1 (80, 25, 120, 2)
    print(field[f], dm.X0[f, indexes].shape)  # 风速 风向 风场1 (80, 25, 120, 2)
    print(field[f], dm.W[f, indexes].shape)  # 对应的气象数据（风速风向，未归一化）  风场1 (80, 14, 2)，共13个小时的数据，也就是14个值，前12个值是过去的，后两个是未来的
    print(field[f], dm.Y0[f, indexes].shape)  # 输出的风向风速   风场1 (80, 25, 20, 2)

print("-----------------------根据验证集，用随机方式生成预测结果，并可视化----------------------------")
#生成的预测结果与验证集的真实数据，做可视化对比（数据来源与  预测值dev_pred.csv和真实值dev_true.csv)，并计算分数，都在这一段
#生成验证集结果并保存，用mean+std的方式
t = 0    # 第 t 个验证集，不要超过你设定的最大值
df = pd.DataFrame(
    data = np.array([
        [*x0, x1, x2, x3, x4] for x0, x1, x2, x3, x4 in
            itertools.product(
                np.vstack([list(itertools.product([field[f]], machine[f])) for f in range(field_len)]).tolist(),    # '风场', '风机'
                period,                     # '时段'
                np.arange(1, 20+1) * 30,    # '时刻'
                [None],                     # '风速'
                [None]                      # '风向'
            )
    ]),
    columns = ['风场', '风机', '时段', '时刻', '风速', '风向']
)
val_pred = df.copy()
all = []
for f in range(field_len):
    indexes = dm.val_indexes[field[f]][t]
    print(indexes)
    for m in range(machine_len):
        for i in indexes:
            #print(field[f],i/24,machine[f][m], dm.X0[f,i,m])
            #print(field[f],int(i/24),machine[f][m],  np.mean(dm.X0[f,i,m], axis=0),np.std(dm.X0[f,i,m], axis = 0))
            one_std=np.random.random((20, 2))*np.std(dm.X0[f,i,m], axis = 0)
            one_mean=np.ones((20,2),dtype = np.int)*np.mean(dm.X0[f,i,m], axis=0)
            #print(one_std+one_mean)
            all = np.append(all,one_std+one_mean).reshape(-1, 2)
val_pred.loc[:, ['风速', '风向']] = all
val_pred["风速"] = val_pred["风速"].astype('float64')
val_pred["风向"] = val_pred["风向"].astype('float64')
val_pred.to_csv('./data/dev_pred.csv', float_format='%.4f', index=False, encoding='utf-8')  #生成随机结果，写入验证集预测文件
logging('Score', cal_R())        # 计算得分
Compare_dev_result(dm,paras.save) # 可视化比较

"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
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

logging('Finished: best dev error')