#include<stdlib.h>
#include<math.h>
#include<stdio.h>

// easy part

/// @brief the forward step of a full connect layer
/// @param x the input of the FC layer in shape (N, K)
/// @param w the parameter of the FC layer in shape (K, M)
/// @param bias the bias of the FC layer in shape (M, )
/// @return the output in shape (N, M)
float* full_connect_forward(float *const x, float *const w, float *const bias, int N, int K, int M) {
    // multiplication of two matrix actually
    float *const res = (float *const) malloc(sizeof(float) * N * M);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            res[i*M + j] = bias[j];
            for (int k = 0; k < K; k++) {
                res[i*M + j] += x[i*K + k] * w[k*M + j];
            }
        }
    }
    return res;
}

/// @brief the forward step of ReLU activate function
/// @param x the input in shape (N, )
/// @return the output in shape (N, )
float* ReLU_forward(float *const x, int N) {
    float *const res = (float *const) malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
        if (x[i] < 0) res[i] = 0;
        else res[i] = x[i];
    }
    return res;
}

/// @brief the backward step of sigmoid activate function
/// @param x the input in shape (N, )
/// @return the output in shape (N, )
float* sigmoid_forward(float *const x, int N) {
    float *const res = (float *const) malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
        res[i] = 1. / (1. + exp(-x[i]));
    }
    return res;
}

/// @brief first calculate the softmax of x and then use the result to calculate cross entropy loss
/// @param a the output or y_hat in shape (N, M)
/// @param y the label or actually y in shape (N, M)
/// @param N the number of batches
/// @param M the number of output
/// @return the loss
float softmax_cross_entropy_loss(float *const a, float *const y, int N, int M) {
    float loss = 0;
    for (int s = 0; s < N; s++) {
        float maxx = a[s*M + 0];
        for (int i = 1; i < N; i++) {
            if (a[s*M + i] > maxx) {
                maxx = a[s*M + i];
            }
        }
        float *x = (float *) malloc(sizeof(float) * N);
        float sumexp = 0;
        for (int i = 0; i < N; i++) {
            x[i] = exp(a[s*M + i] - maxx);
            sumexp += x[i];
        }
        for (int i = 0; i < N; i++) {
            x[i] = x[i] / sumexp;
            loss += -y[i] * log(x[i]);
        }
    }
    loss /= N;
    return loss;
}

// hard part

#define x(s, i, j, k) (x[(s)*C*H*W + (i)*H*W + (j)*W + (k)])
#define w(s, i, j, k) (w[(s)*C*K*K + (i)*K*K + (j)*K + (k)])
#define res(s, i, j, k) (res[(s)*O*outH*outW + (i)*outH*outW + (j)*outW + (k)])

/// @brief Do Conv2d (no padding, stride=1) with (N, C, H, W) input and (O, C, K, K) kernal, and return (N, O, H', W') output
/// @param x the input in shape (N, C, H, W)
/// @param w the parameters of the kernal in shape (O, C, K, K)
/// @param bias the bias of Conv2d layer with chape (O, )
/// @param stride the stride of convolution
/// @param N the number of batches
/// @param C the number of input channels
/// @param H,W the hight and width of input
/// @param O the number of out channels
/// @param K the hight and width of the kernel
/// @return the output in shape (N, H, W, C)
float* convolution_forward(float *const x, float *const w, float *const bias, int stride, int N, int C, int H, int W, int O, int K) {

    int outH = (H - K) / stride + 1, outW = (W - K) / stride + 1;
    float *const res = (float *const) malloc(sizeof(float) * N * O * outH * outW);
    for (int s = 0; s < N; s++) {
        for (int CO = 0; CO < O; CO++) {
            for (int i = 0; i < outH; i++) {
                for (int j = 0; j < outW; j++) {
                    float sum = bias[CO];
                    for (int CI = 0; CI < C; CI++) {
                        for (int k = 0; k < K; k++) {
                            for (int l = 0; l < K; l++) {
                                sum += x(s, CI, i * stride + k, j * stride + l) * w(CO, CI, k, l);
                            }
                        }
                    }
                    res(s, CO, i, j) = sum;
                }
            }
        }
    }
    return res;
}

/// @brief Do AvgPool2d (no padding, stride=K) with (N, C, H, W) input and kernal size (K, K).
/// @param x the input in shape (N, C, H, W)
/// @param K the kernal size and stride as well
/// @return the output of the pooling layer in shape (N, C, H/K, W/K)
float* avg_pool_forward(float *const x, int N, int C, int H, int W, int K) {
    int outH = H / K, outW = W / K, O = C;
    float *const res = (float *const) malloc(sizeof(float) * N * C * outH * outW);
    for (int s = 0; s < N; s++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < outH; i++) {
                for (int j = 0; j < outW; j++) {
                    float sum = 0;
                    for (int k = i * K; k < (i + 1) * K; k++) {
                        for (int l = j * K; l < (j + 1) * K; l++) {
                            sum += x(s, c, k, l);
                        }
                    }
                    sum /= K * K;
                    res(s, c, i, j) = sum;
                }
            }
        }
    }
    return res;
}