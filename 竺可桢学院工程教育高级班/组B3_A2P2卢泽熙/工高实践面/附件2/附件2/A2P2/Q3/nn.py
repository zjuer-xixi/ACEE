import numpy as np

"""
This file consists of all the layer you may use when construct
the Neural Network. Your can choose to follow the comments to
complete this file. OR, you can build the layer by yourself!
"""

# Copy what you have done in Question 2
class Conv2d:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dtype = None):
        pass

    def forward(self, x):
        pass

    def backward(self, dy, lr):
        pass

class ReLU:
    def forward(self, x):
        # Copy what you have done in Question 1
        pass

    def backward(self, dy):
        # TODO: return the gradient of x
        pass

class Tanh:
    def forward(self, x):
        ex = np.exp(x)
        e_x = np.exp(-x)
        self.y = (ex - e_x) / (ex + e_x)
        return self.y

    def backward(self, dy):
        # TODO: return the gradient of x
        pass


class Sigmoid:
    def forward(self, x):
        # Copy what you have done in Question 1
        pass

    def backward(self, dy):
        # TODO: return the gradient of x
        pass

"""
You can complete the layer of MaxPool2d by referring to the implement of
AvgPool2d below, or you can complete this by yourself and improve the
implement of AvgPool2d in the same way.
"""
class MaxPool2d:
    def __init__(self, kernel_size: int, stride = None, padding = 0):
        if stride == None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """
        x - shape (N, C, H, W)
        return the result of MaxPool2d with shape (N, C, H', W')
        """
        # TODO: the forward of MaxPool2d layer

    def backward(self, dy):
        """
        dy - shape (N, C, H', W')
        return the result of gradient dx with shape (N, C, H, W)
        """
        # TODO: the back propogation of MaxPool2d layer


"""
Improve your code in Question 1 (no padding, stride=K) to complete the
forward function of AvgPool2d. You can try to vectorized the backward
function further.
"""
class AvgPool2d:
    def __init__(self, kernel_size: int, stride = None, padding = 0):
        if stride == None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """
        x - shape (N, C, H, W)
        return the result of AvgPool2d with shape (N, C, H', W')
        """
        # TODO: the forward of AvgPool2d layer
        res = ...

        return res

    def backward(self, dy):
        """
        dy - shape (N, C, H', W')
        return the result of gradient dx with shape (N, C, H, W)
        """
        N, C, H, W = self.x.shape
        H_pad, W_pad = H + self.padding * 2, W + self.padding * 2
        K, stride = self.kernel_size, self.stride
        outH, outW = (H_pad - K) // stride + 1, (W_pad - K) // stride + 1
        if outH != dy.shape[2] or outW != dy.shape[3]:
            raise RuntimeError
        dx = np.zeros((N, C, H_pad, W_pad))
        for i in range(outH):
            for j in range(outW):
                dx[:, :, i*stride:i*stride+K, j*stride:j*stride+K] += dy[:, :, i:i+1, j:j+1] / K**2
        return dx[:, :, self.padding:self.padding+H, self.padding:self.padding+W]

        

class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(in_features, out_features) * np.sqrt(2 / self.in_features)
        self.w_grad = np.zeros((in_features, out_features))
        if bias:
            self.bias = np.random.randn(out_features)
            self.b_grad = np.zeros(out_features)
        else:
            self.bias = None
        

    def forward(self, x):
        """
        x - shape (N, C)
        return the result of Linear layer with shape (N, O)
        """
        self.x = x
        x = x.reshape(x.shape[0], -1)
        if x.shape[1] != self.in_features:
            raise RuntimeError
        res = np.matmul(x, self.weight)
        # print(res.shape, self.bias.shape)
        if self.bias is not None:
            res += self.bias
        return res


    def backward(self, dy, lr):
        """
        dy - shape (N, O)
        return the result of gradient dx with shape (N, C)
        """
        N, O = dy.shape
        if N != self.x.shape[0] or O != self.out_features:
            raise RuntimeError
        x = self.x.reshape(self.x.shape[0], -1)
        self.w_grad = np.matmul(x.T, dy) / N
        if self.bias is not None:
            self.b_grad = np.sum(dy, axis=0) / N

        dx = np.matmul(dy, self.weight.T).reshape(*self.x.shape)

        self.weight = self.weight - lr * self.w_grad
        if self.bias is not None:
            self.bias = self.bias - lr * self.b_grad
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
            z[np.indices((N,)), label.reshape(N,)] = 1
            label = z
        elif label.shape[1] != M:
            raise RuntimeError
        
        # Copy what you have done in Question 2
        loss = ...
        acc = ...
        dx = ...

        return loss, acc, dx.reshape(*dx_shape)