import numpy as np

"""
This is the implement of Conv2d layer.
Your task is to complete the forward and backward part of this layer
and try to vectorize them as possible.
"""
class Conv2d:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, bias: bool = True):
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
        """
        x - shape (N, C, H, W)
        return the result of Conv2d with shape (N, O, H', W')
        """
        self.x = x

        # TODO: padding the x

        N, C, H, W = x.shape
        if C != self.in_channels:
            raise RuntimeError(f'Expected input{x.shape} to have {self.in_channels} \
channels, but got {C} channels instead.')
        O, C, K, K = self.weight.shape
        outH, outW = (H - K) // self.stride + 1, (W - K) // self.stride + 1

        # TODO: Do Conv2d to get result
        res = ...
        
        return res

    def backward(self, dy, lr):
        """
        dy - the gradient of last layer with shape (N, O, H', W')
        lr - learning rate
        calculate self.w_grad to update self.weight,
        calculate self.b_grad to update self.bias,
        return the result of gradient dx with shape (N, C, H, W)
        """
        
        N, C, H, W = self.x.shape
        N, O, outH, outW = dy.shape
        if O != self.out_channels:
            raise RuntimeError(f'Expected input{dy.shape} to have {self.out_channels} \
channels, but got {O} channels instead.')
        O, C, K = self.out_channels, self.in_channels, self.kernel_size

        # TODO: filling 0 to dy if stride > 1 to ensure correctness
        if self.stride > 1:
            dy = ...

        # TODO: padding the x
        x = ...
        N, C, H_pad, W_pad = x.shape
        dH, dW = (H_pad - K) % self.stride, (W_pad - K) % self.stride

        # cut off the entry with no contribution
        H_fact, W_fact = H_pad - dH, W_pad - dW
        x = x[:, :, :H_fact, :W_fact]

        # TODO: calculate self.w_grad and self.b_grad
        self.b_grad = ...
        self.w_grad = ...

        dx = np.zeros((N, C, H, W))
        # TODO: calculate dx
        dx = ...

        # TODO: update the parameter
        self.bias = self.bias - lr * self.b_grad
        self.weight = self.weight - lr * self.w_grad

        return dx

"""
This is the implement of the loss class
Implement and vectorized it as possible!
"""
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
        
        # TODO: calculate the softmax and return loss, accuracy and the gradient of x
        # Think the softmax output as the possibility of each classification.
        # You can take the possibility of the correct class as accuracy.
        loss = ...
        acc = ...
        dx = ...

        return loss, acc, dx.reshape(*dx_shape)