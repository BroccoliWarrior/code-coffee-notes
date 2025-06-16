# -*- coding:UTF-8 -*-
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import os

# 定义LSTM神经网络
class LstmRNN(nn.Module):
    """
        参数：
        - input_size: 特征维度
        - hidden_size: 隐藏单元数量
        - output_size: 输出维度
        - num_layers: LSTM堆叠层数
    """
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) # 利用torch.nn中的LSTM模型
        self.forwardCalculation = nn.Linear(hidden_size, output_size)
 
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x是输入，大小为(seq_len, batch, input_size)
        s, b, h = x.shape  # x是输出，大小为(seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x

if __name__ == '__main__':
    # 创建保存目录
    os.makedirs('models/task8', exist_ok=True)
    os.makedirs('result/task8', exist_ok=True)
    
    # 创建数据集
    data_len = 200
    t = np.linspace(0, 12*np.pi, data_len)
    sin_t = np.sin(t)
    cos_t = np.cos(t)

    dataset = np.zeros((data_len, 2))
    dataset[:,0] = sin_t
    dataset[:,1] = cos_t
    dataset = dataset.astype('float32')

    # 绘制部分原始数据集
    plt.figure()
    plt.plot(t[0:60], dataset[0:60,0], label='sin(t)')
    plt.plot(t[0:60], dataset[0:60,1], label = 'cos(t)')
    plt.plot([2.5, 2.5], [-1.3, 0.55], 'r--', label='t = 2.5') # t = 2.5
    plt.plot([6.8, 6.8], [-1.3, 0.85], 'm--', label='t = 6.8') # t = 6.8
    plt.xlabel('t')
    plt.ylim(-1.2, 1.2)
    plt.ylabel('sin(t) and cos(t)')
    plt.legend(loc='upper right')

    # 选择训练和测试数据集
    train_data_ratio = 0.5 # 选择50%的数据用于训练
    train_data_len = int(data_len*train_data_ratio)
    train_x = dataset[:train_data_len, 0]
    train_y = dataset[:train_data_len, 1]
    INPUT_FEATURES_NUM = 1
    OUTPUT_FEATURES_NUM = 1
    t_for_training = t[:train_data_len]

    # test_x = train_x
    # test_y = train_y
    test_x = dataset[train_data_len:, 0]
    test_y = dataset[train_data_len:, 1]
    t_for_testing = t[train_data_len:]

    # ----------------- 训练 -------------------
    train_x_tensor = train_x.reshape(-1, 5, INPUT_FEATURES_NUM) # 设置批次大小为5
    train_y_tensor = train_y.reshape(-1, 5, OUTPUT_FEATURES_NUM) # 设置批次大小为5
 
    # 将数据转换为pytorch张量
    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_tensor = torch.from_numpy(train_y_tensor)
    # test_x_tensor = torch.from_numpy(test_x)
 
    lstm_model = LstmRNN(INPUT_FEATURES_NUM, 16, output_size=OUTPUT_FEATURES_NUM, num_layers=1) # 16个隐藏单元
    print('LSTM模型:', lstm_model)
    print('模型参数:', lstm_model.parameters)
 
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)
 
    max_epochs = 10000
    for epoch in range(max_epochs):
        output = lstm_model(train_x_tensor)
        loss = loss_function(output, train_y_tensor)
 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
 
        if loss.item() < 1e-4:
            print('训练轮次 [{}/{}], 损失: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
            print("达到目标损失值")
            break
        elif (epoch+1) % 100 == 0:
            print('训练轮次: [{}/{}], 损失:{:.5f}'.format(epoch+1, max_epochs, loss.item()))
 
    # 在训练数据集上进行预测
    predictive_y_for_training = lstm_model(train_x_tensor)
    predictive_y_for_training = predictive_y_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    torch.save(lstm_model.state_dict(), 'models/task8/model_params.pkl') # 保存模型参数到文件
 
    # ----------------- 测试 -------------------
    # lstm_model.load_state_dict(torch.load('models/task8/model_params.pkl'))  # 从文件加载模型参数
    lstm_model = lstm_model.eval() # 切换到测试模式

    # 在测试数据集上进行预测
    test_x_tensor = test_x.reshape(-1, 5, INPUT_FEATURES_NUM) # 设置批次大小为5，与训练集相同
    test_x_tensor = torch.from_numpy(test_x_tensor)
 
    predictive_y_for_testing = lstm_model(test_x_tensor)
    predictive_y_for_testing = predictive_y_for_testing.view(-1, OUTPUT_FEATURES_NUM).data.numpy()
 
    # ----------------- 绘图 -------------------
    plt.figure()
    plt.plot(t_for_training, train_x, 'g', label='sin_trn')
    plt.plot(t_for_training, train_y, 'b', label='ref_cos_trn')
    plt.plot(t_for_training, predictive_y_for_training, 'y--', label='pre_cos_trn')

    plt.plot(t_for_testing, test_x, 'c', label='sin_tst')
    plt.plot(t_for_testing, test_y, 'k', label='ref_cos_tst')
    plt.plot(t_for_testing, predictive_y_for_testing, 'm--', label='pre_cos_tst')

    plt.plot([t[train_data_len], t[train_data_len]], [-1.2, 4.0], 'r--', label='separation line') # 分隔线

    plt.xlabel('t')
    plt.ylabel('sin(t) and cos(t)')
    plt.xlim(t[0], t[-1])
    plt.ylim(-1.2, 4)
    plt.legend(loc='upper right')
    plt.text(14, 2, "train", size = 15, alpha = 1.0)
    plt.text(20, 2, "test", size = 15, alpha = 1.0)

    # 保存图片到result/task8目录
    plt.savefig('result/task8/lstm_sincos_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()