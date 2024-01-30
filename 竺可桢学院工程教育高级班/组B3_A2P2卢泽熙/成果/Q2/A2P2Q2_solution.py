# -*- codeing = utf-8 -*-
# @Time : 2023/4/1 11:05
import numpy as np


class Conv2d:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        self.x = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        O, C, K = out_channels, in_channels, kernel_size
        self.weight = np.random.randn(O, C, K, K) * np.sqrt(2 / (C * K * K))
        self.w_grad = np.zeros((O, C, K, K))

        if bias:
            self.bias = np.random.randn(self.out_channels)
            self.b_grad = np.zeros(self.out_channels)
        else:
            self.bias = None

    def forward(self, x):

        self.x = x

        if self.padding != 0:
            x = np.pad(x, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], 0)
        N, C, H, W = x.shape
        if C != self.in_channels:
            raise RuntimeError(f'Expected input{x.shape} to have {self.in_channels} \
            channels, but got {C} channels instead.')
        O, C, K, K = self.weight.shape
        outH, outW = (H - K) // self.stride + 1, (W - K) // self.stride + 1
        # 正向传播进行卷积操作
        res = np.zeros((N, O, outH, outW))
        for s in range(N):
            for CO in range(O):
                for i in range(outH):
                    for j in range(outW):
                        sum = self.bias[CO]
                        for CI in range(C):
                            for k in range(K):
                                for l in range(K):
                                    sum += x[s, CI, i * self.stride + k, j * self.stride + l] * self.weight[CO, CI, k, l]
                        res[s, CO, i, j] = sum

        return res

    def backward(self, dy, lr):

        N, C, H, W = self.x.shape
        N, O, outH, outW = dy.shape
        if O != self.out_channels:
            raise RuntimeError(f'Expected input{dy.shape} to have {self.out_channels} \
channels, but got {O} channels instead.')
        O, C, K = self.out_channels, self.in_channels, self.kernel_size

        # TODO: filling 0 to dy if stride > 1 to ensure correctness
        if self.stride > 1:
            dy = np.zeros((N, O, H + K - 1, W + K - 1))
            # 保持N、O不变，对dy初始化填充后按步长间隔重新赋值
            dy[:, :, ::self.stride, ::self.stride] = self.dy

        # TODO: padding the x
        # 对x完成填充操作
        x = np.pad(self.x, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], 0)
        N, C, H_pad, W_pad = x.shape
        dH, dW = (H_pad - K) % self.stride, (W_pad - K) % self.stride

        # cut off the entry with no contribution
        H_fact, W_fact = H_pad - dH, W_pad - dW
        x = x[:, :, :H_fact, :W_fact]

        # TODO: calculate self.w_grad and self.b_grad
        # 计算偏置梯度
        self.b_grad = dy.sum(axis=(0, 2, 3))
        # 计算卷积核的梯度
        self.w_grad = np.dot(dy, x)

        dx = np.zeros((N, C, H, W))
        # TODO: calculate dx
        # dx部分的代码因水平原因无法补全

        # TODO: update the parameter
        # lr对应学习率
        if self.bias is not None:
            self.bias = self.bias - lr * self.b_grad
        self.weight = self.weight - lr * self.w_grad

        return dx


class CrossEntropyLoss:
    def __call__(self, x, label):
        dx_shape = x.shape
        if len(x.shape) != 2:
            x = x.reshape(x.shape[0], -1)
        if len(label.shape) != 2:
            label = label.reshape(label.shape[0], -1)
        N, M = x.shape
        if label.shape[0] != N:
            raise RuntimeError
        elif label.shape[1] == 1:
            z = np.zeros((N, M))
            z[np.indices((N,)), label.reshape(N, )] = 1
            label = z
        elif label.shape[1] != M:
            raise RuntimeError

        # TODO: calculate the softmax and return loss, accuracy and the gradient of x
        # 找出输入矩阵x每一行的最大值
        maxx = np.max(x, axis=1, keepdims=True)
        # 得到每个分类的概率值(使用了 maxx 变量来缩放输入矩阵 x，以防止指数函数的值溢出)
        y_hat = np.exp(x - maxx) / np.sum(np.exp(x - maxx), axis=1, keepdims=True)
        # 计算损失函数的值
        loss = -np.sum(label * np.log(y_hat)) / N
        # 使用 np.argmax(y_hat, axis=1) 计算出每个样本在预测值 y_hat 中概率最大的类别的索引。
        # 然后使用 np.argmax(label, axis=1) 计算出每个样本在标签 label 中所属的类别的索引。如果这两个索引相等，则说明该样本被正确分类。
        acc = np.mean(np.argmax(y_hat, axis=1) == np.argmax(label, axis=1))
        # 利用公式计算x的梯度，这里用到了广播机制
        dx = -label * (1 - y_hat) / N

        return loss, acc, dx.reshape(*dx_shape)


