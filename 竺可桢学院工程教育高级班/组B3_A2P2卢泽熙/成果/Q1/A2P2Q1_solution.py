# -*- codeing = utf-8 -*-
# @Time : 2023/4/1 9:02
# 导入numpy库
import numpy as np


# 定义一个前向传播函数，x为输入数据，w为权重系数，bias为偏置
def full_connect_forward(x, w, bias, N, K, M):
    # 对res进行初始化
    res = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            res[i, j] = bias[j]
            for k in range(K):
                # 每个特征乘以相应权重系数作为输出
                res[i, j] += x[i, k] * w[k, j]
    return res


# 定义一个激活函数
def ReLU_forward(x, N):
    # 对res进行初始化
    res = np.zeros((N,))
    # ReLU的特征是，当输入小于零，输出为零；当输入大于零，输出等于输入
    for i in range(N):
        if x[i] < 0:
            res[i] = 0
        else:
            res[i] = x[i]
    return res


# 定义另一种形式的激活函数
def sigmoid_forward(x, N):
    # 对res进行初始化
    res = np.zeros((N,))
    # sigmoid函数可保证输出在（0，1）之间
    for i in range(N):
        res[i] = 1. / (1. + np.exp(-x[i]))
    return res


# 定义交叉熵损失函数
def softmax_cross_entropy_loss(a, y, N, M):
    loss = 0.0
    for s in range(N):
        # 找出每一段的最大值，以进行后续处理
        maxx = a[s * M + 0]
        for i in range(1, M):
            if a[s * M + i] > maxx:
                maxx = a[s * M + i]
        # 初始化x
        x = np.zeros((M,))
        sum = 0
        # 根据公式计算loss值
        for i in range(M):
            x[i] = np.exp(a[s * M + i] - maxx)
            sum += x[i]
        for i in range(M):
            x[i] /= sum
            loss += -y[i] * np.log(x[i])
    loss /= N
    return loss


# @brief 对具有(N, C, H, W)形状的输入和(O, C, K, K)卷积核进行Conv2d操作（无填充，stride=1），并返回(N, O, H', W')形状的输出
# @param x 形状为(N, C, H, W)的输入
# @param w 形状为(O, C, K, K)的卷积核参数
# @param bias Conv2d层的偏置，形状为(O, )
# @param stride 卷积的步幅
# @param N 批量大小
# @param C 输入通道数
# @param H,W 输入的高度和宽度
# @param O 输出通道数
# @param K 卷积核的高度和宽度
# @return 形状为(N, H, W, C)的输出
# 卷积神经网络的前向传播函数，stride为步长
def convolution_forward(x, w, bias, stride, N, C, H, W, O, K):
    # 输出数据的out_size公式为[(in_size+2padding-K)/stride +1],此题中padding为0
    outH = (H - K) // stride + 1
    outW = (W - K) // stride + 1
    res = np.zeros((N, O, outH, outW))
    # 进行卷积操作
    for s in range(N):
        for CO in range(O):
            for i in range(outH):
                for j in range(outW):
                    sum = bias[CO]
                    for CI in range(C):
                        for k in range(K):
                            for l in range(K):
                                sum += x[s, CI, i * stride + k, j * stride + l] * w[CO, CI, k, l]
                    res[s, CO, i, j] = sum

    return res


# 定义一个平均池化函数，x为输入张量，形状为 (N, C, H, W);K为池化核大小；池化层的输出张量形状为 (N, C, H/K, W/K)
def avg_pool_forward(x, N, C, H, W, K):
    # 此处对输出size的计算运用到了前面的公式
    outH = H // K
    outW = W // K
    O = C
    res = np.zeros((N, C, outH, outW))
    # 多重循坏遍历数据完成操作
    for s in range(N):
        for c in range(C):
            for i in range(outH):
                for j in range(outW):
                    sum = 0
                    for k in range(i * K, (i * K + K)):
                        for l in range(j * K, (j * K + K)):
                            # 将池化核覆盖的数据相加
                            sum += x[s, c, k, l]
                    # 平均化操作
                    res[s, c, i, j] = sum / (K * K)
    return res


# 对此函数向量化
def full_connect_forward_vectorized(x, w, bias):
    # 此处用到了广播机制
    res = np.dot(x, w) + bias
    return res


def ReLU_forward_vectorized(x):
    # 向量化省略了for循环
    res = np.maximum(0, x)
    return res


def sigmoid_forward_vectorized(x):
    # 向量化省略了for循环
    res = 1. / (1. + np.exp(-x))
    return res


def softmax_cross_entropy_loss_vectorized(a, y, N, M):
    # 将 ⃗x 中的每个元素都减去 ⃗x 中元素的最大值。
    maxx = np.max(a, axis=1)
    sum = np.sum(np.exp(a - maxx), axis=1)
    # 完成归一化操作
    res = np.exp(a - maxx) / sum
    loss = -np.sum(y * np.log(res)) / N
    return loss


def convolution_forward_vectorized(x, w, bias, stride, N, C, H, W, O, K):
    outH = (H - K) // stride + 1
    outW = (W - K) // stride + 1
    res = np.zeros((N, O, outH, outW))
    # 使用三重循环遍历每个输入数据的样本
    for s in range(N):
        for i in range(outH):
            for j in range(outW):
                # 从输入数据中提取出一个切片x_slice，然后将其展平为一个向量
                x_slice = x[s, :, i * stride:i * stride + K, j * stride:j * stride + K]
                x_slice = x_slice.reshape(C, -1)
                # 将卷积核w展平为一个矩阵
                w_flat = w.reshape(O, -1)
                # 使用np.dot函数进行矩阵乘法
                sum = np.dot(x_slice, w_flat.T) + bias
                res[s, :, i, j] = sum

    return res


def avg_pool_forward(x, N, C, H, W, K):

    outH = H // K
    outW = W // K
    res = np.zeros((N, C, outH, outW))
    # reshape输入数据，方便使用sum函数
    x_reshaped = x.reshape(N, C, outH, K, outW, K)
    # 沿着指定轴计算元素和
    sum = x_reshaped.sum(axis=(3, 5))
    # 平均化操作
    res = sum / (K * K)
    return res
