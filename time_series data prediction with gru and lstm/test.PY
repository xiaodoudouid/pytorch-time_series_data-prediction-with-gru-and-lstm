'''test完整模块'''

# 用户：Ejemplarr
# 编写时间:2022/3/24 22:10
from train import device
from data_preparation import data_start, data_prediction_to_f_and_t, dataset_split_4sets, lengths, targets
from GRU import GRU
from LSTM import LSTM

import torch
import matplotlib.pyplot as plt
import numpy as np

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的计算设备: {device}")

# 1. 获取数据和 Scaler
x_raw, y_scaled_tensor, scaler = data_start()

x = x_raw
# scaler 要求输入必须是二维的 (N, 1)，否则会报 ValueError: Expected 2D array
y = scaler.inverse_transform(y_scaled_tensor.cpu().numpy().reshape(-1, 1))

dataset_features, dataset_target = data_prediction_to_f_and_t(y_scaled_tensor, lengths, targets)
# 切分出测试集
_, _, test_features, test_target = dataset_split_4sets(dataset_features, dataset_target)

# 导入保存好的网络
net_gru = GRU().to(device)
net_gru.load_state_dict(torch.load('gru.pt'))
net_lstm = LSTM().to(device)
net_lstm.load_state_dict(torch.load('lstm.pt'))


def y_pred_to_numpy(y_pred):
    '''
    :param y_pred: 网络的输出 (Tensor)
    :return: 一个反归一化后的 numpy 数组
    '''
    # === Tensor 转 Numpy 并调整形状 ===
    y_pred_np = y_pred.detach().cpu().numpy()

    # 同样为了防止报错，强制转为二维 (N, 1) 再进行反归一化
    y_pred_np = y_pred_np.reshape(-1, 1)

    y_pred_real = scaler.inverse_transform(y_pred_np)

    return y_pred_real


# 定义测试函数
def test_for_gru(dataset_features):
    dataset_features = dataset_features.reshape([len(dataset_features), lengths, 1])

    # 模型预测
    y_pred = net_gru(torch.from_numpy(dataset_features).float().to(device))

    # 数据转换
    y_pred = y_pred_to_numpy(y_pred)

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    # 这里的切片逻辑：x轴对应从第 lengths 个点开始，长度等于预测数据的长度
    plt.plot(x[lengths: lengths + len(y_pred)], y_pred)
    plt.legend(('data', 'data_pred:{}'.format(targets)), loc='upper right')
    plt.title('GRU Prediction')
    plt.show()


def test_for_lstm(dataset_features):
    dataset_features = dataset_features.reshape([len(dataset_features), lengths, 1])

    y_pred = net_lstm(torch.from_numpy(dataset_features).float().to(device))
    y_pred = y_pred_to_numpy(y_pred)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.plot(x[lengths: lengths + len(y_pred)], y_pred)
    plt.legend(('data', 'data_pred:{}'.format(targets)), loc='upper right')
    plt.title('LSTM Prediction')
    plt.show()


if __name__ == '__main__':
    # 1. 先画原始数据图
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.title('Original Data')
    plt.show()

    # 2. 进行 GRU 测试
    test_for_gru(dataset_features)

    # 3. 进行 LSTM 测试
    test_for_lstm(dataset_features)