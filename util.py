# 画图函数
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
plt.switch_backend('agg')
import itertools
import time
from deal_dataset import *


class SimpleLogger(object):
    def __init__(self, f, header='#logger output'):
        dir = os.path.dirname(f)
        #print('test dir', dir, 'from', f)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(f, 'w',encoding='utf-8') as fID:
            fID.write('%s\n'%header)
        self.f = f

    def __call__(self, *args):
        #standard output
        print(*args)
        #log to file
        try:
            with open(self.f, 'a',encoding='utf-8') as fID:
                fID.write(' '.join(str(a) for a in args)+'\n')
        except:
            print('Warning: could not log to', self.f)


def visualize_prediction_compare(X, Y, Y_pre=None, dir_name='visualizations',png_name='name'):
    plt.figure(figsize=(15, 8))
    outputs_names = ['wind speed', 'wind direction']
    X_length = np.arange(0, 120)
    Y_length = np.arange(120, 140)
    for i, y_name in enumerate(outputs_names):
        y_pred_i = Y_pre[:, i]
        x_i = X[:, i]
        y_i = Y[:, i]
        plt.subplot(2, 1, i + 1)
        plt.ylabel(y_name)
        plt.xlabel('Time(30s)')
        plt.plot(X_length, x_i, '-k', label='X')
        plt.plot(Y_length, y_i, '-b', label='Y')
        plt.plot(Y_length, y_pred_i, '-r', label='Y_pre')

    plt.legend()
    os.makedirs(dir_name.encode('utf-8'))
    file = str((dir_name + '.png').encode('utf-8'),encoding = "utf-8")
    ff = open(file.encode('utf-8'), mode='w')
    plt.savefig(ff)
    plt.close()


def standardize(val_df):
    """
    记住，位置固定，顺序已知。
    不管你是有缺失值还是顺序不对，这段代码都帮你解决。
    """
    field = np.array([
        '风场1',
        '风场2'
    ])
    field_len = field.shape[0]

    machine = np.array([
        [f'x{i}' for i in range(26, 50 + 1)],
        [f'x{i}' for i in range(25, 49 + 1)]
    ])
    machine_len = machine[0].shape[0]

    season = np.array(['春', '夏', '秋', '冬'])
    season_len = season.shape[0]

    period = np.array([f'{s}_{str(i).zfill(2)}' for s in season for i in range(1, 20 + 1)])
    period_len = period.shape[0]

    df = pd.DataFrame(
        data=np.array([
            [*x0, x1, x2, x3, x4] for x0, x1, x2, x3, x4 in
            itertools.product(
                np.vstack([list(itertools.product([field[f]], machine[f])) for f in range(field_len)]).tolist(),
                # '风场', '风机'
                period,  # '时段'
                np.arange(1, 20 + 1) * 30,  # '时刻'
                [None],  # '风速'
                [None]  # '风向'
            )
        ]),
        columns=['风场', '风机', '时段', '时刻', '风速', '风向']
    )

    return pd.merge(
        left=df[['风场', '风机', '时段', '时刻']].copy(),
        right=val_df,
        how='left',
        on=['风场', '风机', '时段', '时刻']
    ).fillna(0)
# 计算分数
def cal_R():
    val_true = standardize(pd.read_csv('./data/dev_true.csv', encoding='utf-8'))  #验证集真实值
    val_pred = standardize(pd.read_csv('./data/dev_pred.csv', encoding='utf-8'))  #验证集预测值

    # 公式 6
    w = np.arange(2 * 25 * 80 * 20, dtype=float) % 20
    w[w < 10] = 0.06
    w[w >= 10] = 0.04

    # 公式 3
    m = (w * np.abs(val_pred['风速'] - val_true['风速']).values).reshape(2 * 25 * 80, 20).sum(-1)

    # 公式 7
    threshold = np.repeat([0.15, 0.086], 25 * 80 * 20)
    a = w * (val_true['风速'].values > threshold)
    # 公式 5
    e = np.minimum(np.mod((val_pred['风向'] - val_true['风向']).values, 1),
                   np.mod((val_true['风向'] - val_pred['风向']).values, 1))
    # 公式 4
    n = (a * e).reshape(2 * 25 * 80, 20).sum(-1)

    # 公式 2
    # 2个风场，每个风场25个风机，每个风机80个时段
    E1, E2 = (0.7 * m + 0.3 * n).reshape(2, 25, 80).mean(1)

    # 公式 1
    R = 100 / (1 + E1.sum() + E2.sum())
    print(f'R: {R:.3f}')



def Compare_dev_result(dm,dir):
    # pd.DataFrame(
    #     data = {field[f]: dm.val_indexes[field[f]][0] for f in range(field_len)},
    #     index = period
    # ).to_csv('val_indexes.csv', float_format='%.4f', encoding='utf-8')
    new_season = np.array(['Spring', 'Summer', 'Autumn', 'Winter'])
    new_period = np.array([f'{s}_{str(i).zfill(2)}' for s in season for i in range(1, 20 + 1)])
    w_df = pd.DataFrame(
        index=np.arange(80 * 14),
        columns = ['时段', '时刻', '风速', '风向']
    )
    w_df.loc[:, ['时段', '时刻']] = np.array(list(itertools.product(period, np.arange(-11, 2+1))))

    df = pd.DataFrame(
        data = {'time': np.arange(1, 121) * 30},
        columns = ['time', '变频器电网侧有功功率', '外界温度', '风速', '风向']
    )

    val_true_df = dm.sample.copy()
    val_pred_df = read_csv('./data/dev_pred.csv') #读取预测结果文件

    var_pred = np.array(val_pred_df[['风速', '风向']])

    #print(var_pred)
    i = -1
    root = dir
    os.makedirs(root, exist_ok=True)
    for f in range(field_len):
        os.makedirs(os.path.join(root, field[f]).encode('utf-8'), exist_ok=True)
        for m in tqdm(range(machine_len)):
            os.makedirs(os.path.join(root, field[f], machine[f][m]).encode('utf-8'), exist_ok=True)
            for p in range(period_len):
                i += 1
                val_df = df.copy()
                val_df.loc[:, ['变频器电网侧有功功率', '外界温度']] = dm.X_[f, dm.val_indexes[field[f]][0][p], m]
                val_df.loc[:, ['风速', '风向']] = dm.X0[f, dm.val_indexes[field[f]][0][p], m]
                #val_df.to_csv(os.path.join(root, field[f], machine[f][m], period[p]) + '.csv', float_format='%.7f', index=False, encoding='utf-8')
                visualize_prediction_compare(dm.X0[f, dm.val_indexes[field[f]][0][p], m],dm.Y0[f, dm.val_indexes[field[f]][0][p], m],Y_pre = var_pred[i*20:(i+1)*20,:],dir_name=os.path.join(dir, field[f], machine[f][m]), png_name=new_period[p])