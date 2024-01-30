# -*- codeing = utf-8 -*-
# @Time : 2023/4/1 9:56
import time
import numpy as np


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


def full_connect_forward_vectorized(x, w, bias):
    # 此处用到了广播机制
    res = np.dot(x, w) + bias
    return res


# 生成输入数据
N = 500
K = 100
M = 50
x = np.random.randn(N, K)
w = np.random.randn(K, M)
bias = np.random.randn(M)

# 测试full_connect_forward
start_time = time.time()
res1 = full_connect_forward(x, w, bias, N, K, M)
end_time = time.time()
print(f"full_connect_forward运行时间：{(end_time - start_time)*1000}ms")

# 测试full_connect_forward_vectorized
start_time = time.time()
res2 = full_connect_forward_vectorized(x, w, bias)
end_time = time.time()
print(f"full_connect_forward_vectorized运行时间：{(end_time - start_time)*1000}ms")


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


def avg_pool_forward_vectorized(x, N, C, H, W, K):
    outH = H // K
    outW = W // K
    O = C
    res = np.zeros((N, C, outH, outW))
    x_reshaped = x.reshape(N, C, outH, K, outW, K)
    # 直接利用向量化操作实现平均计算
    res = np.mean(x_reshaped, axis=(3, 5))
    return res


# 构造输入数据
N, C, H, W, K = 20, 6, 64, 64, 4
x = np.random.rand(N, C, H, W)

# 测试 avg_pool_forward 函数的运行时间
start_time = time.time()
avg_pool_forward(x, N, C, H, W, K)
end_time = time.time()
print(f"avg_pool_forward运行时间：{(end_time - start_time)*1000}ms")

# 测试 avg_pool_forward_vectorized 函数的运行时间
start_time = time.time()
avg_pool_forward_vectorized(x, N, C, H, W, K)
end_time = time.time()
print(f"avg_pool_forward_vectorized运行时间：{(end_time - start_time)*1000}ms")

