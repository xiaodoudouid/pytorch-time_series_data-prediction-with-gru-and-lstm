'''train完整模块 (优化版)'''
# 编写时间:2022/3/24 22:10 (Modified)

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # 用来画Loss图

from GRU import GRU
from LSTM import LSTM

'''
   数据的导入
   可调优数据的定义
   网络实例化
   优化器的定义
   数据搬移至gpu
   损失函数的定义
   开始训练
'''

# 注意这里 data_start 会返回三个值了
from data_preparation import data_start, data_prediction_to_f_and_t, dataset_to_Dataset, dataset_split_4sets, lengths, \
    targets

# 可调参数的定义
BATCH_SIZE = 16
EPOCH = 10
LEARN_RATE = 1e-3

# 数据的导入
x_raw, y, scaler = data_start()  # 接收 scaler
dataset_features, dataset_target = data_prediction_to_f_and_t(y, lengths, targets)
trian_features, train_target, test_features, test_target = dataset_split_4sets(dataset_features, dataset_target)
train_set = dataset_to_Dataset(data_features=trian_features, data_target=train_target)

train_set_iter = DataLoader(dataset=train_set,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            drop_last=True)
#配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的计算设备: {device}")

# 网络的实例化
net_gru = GRU().to(device)
net_lstm = LSTM().to(device)

# 优化器的定义
optim_gru = optim.Adam(params=net_gru.parameters(), lr=LEARN_RATE)
optim_lstm = optim.Adam(params=net_lstm.parameters(), lr=LEARN_RATE)

# 损失函数的定义
loss_fuc = nn.MSELoss()


def train_network(data, device, loss_fuc, net, optim, Epoch, name):
    loss_history = []
    print(f"开始训练 {name} ...")
    for epoch in range(Epoch):
        epoch_loss = []
        for batch_idx, (x, y) in enumerate(data):
            x = x.reshape([BATCH_SIZE, lengths, 1]).to(device)
            y = y.reshape((BATCH_SIZE, targets)).to(device)

            y_pred = net(x)
            loss = loss_fuc(y, y_pred)

            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss.append(loss.item())

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        loss_history.append(avg_loss)
        #if (epoch + 1) % 10 == 0:
        print(f'{name} Epoch {epoch + 1}: loss = {avg_loss:.6f}')
    return loss_history


def main():
    start = time.perf_counter()

    loss_gru = train_network(train_set_iter, device, loss_fuc, net_gru, optim_gru, EPOCH, "GRU")
    loss_lstm = train_network(train_set_iter, device, loss_fuc, net_lstm, optim_lstm, EPOCH, "LSTM")

    end = time.perf_counter()
    print('训练总耗时：{:.2f}s'.format(end - start))

    # 保存模型
    torch.save(net_gru.state_dict(), 'gru.pt')
    torch.save(net_lstm.state_dict(), 'lstm.pt')

    # 简单的 Loss 对比图
    plt.plot(loss_gru, label='GRU Loss')
    plt.plot(loss_lstm, label='LSTM Loss')
    plt.legend()
    plt.title("Training Loss")
    plt.show()


if __name__ == '__main__':
    main()