#include "conv2d.h"
#include <mkl.h>
#include <omp.h>

using namespace lamp;

Conv2d::Conv2d(int input, int output, int kernel, RandomGen& rng, activ_fn activation_fn) : activation_fn(activation_fn),
    k(kernel), stride(1) {
    int* shape = (int*) mkl_malloc(4 * sizeof(int), 32);
    this->filters = Tensor::random(new Shape(output, input, kernel, kernel), rng);
    // this->bias = Tensor::random
}

Conv2d::~Conv2d() {
    delete this->filters;
}

Tensor& Conv2d::im2col(Tensor& x) {
    int N = x.shape->n;
    int C = x.shape->c;
    int H = x.shape->h;
    int W = x.shape->w;

    int out_h = (H - this->k) / this->stride + 1;
    int out_w = (W - this->k) / this->stride + 1;

    float* col_data = (float*) mkl_malloc(N * out_h * out_w * C * this->k * this->k, 32);

    int im2col_idx = 0;
    #pragma omp parallel for simd collapse(6)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int c = 0; c < C; c++) {
                    for (int kh = 0; kh < this->k; kh++) {
                        for (int kw = 0; kw < this->k; kw++) {
                            int x_idx = ((n * C + c) * H + (h * this->stride + kh)) * W + (w * this->stride + kw);
                            *(col_data+im2col_idx) = *(x.data+x_idx);
                            im2col_idx++;
                        }
                    }
                }
            }
        }
    }
    return *(new Tensor(col_data, new Shape(1, 1, N * out_h * out_w, C * this->k * this->k)));
}

Tensor& Conv2d::col2im(Tensor& x) {
    return *filters;
}

Tensor& Conv2d::forward(Tensor& x) {
    Tensor& col = im2col(x);
    this->filters->reshape(1,1,this->out, this->in * this->k * this->k);

    Tensor& out = filters->matmul(x);
    return out;
}

Tensor& Conv2d::sanity_check(Tensor& x) {
    return forward(x);
}

void Conv2d::backward() {

}


/*

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

using namespace std;

// GEMM-based Convolution function following Low-memory Algorithm from arXiv:1709.03395
vector<vector<vector<vector<float>>>> gemm_convolution(
    const vector<vector<vector<vector<float>>>>& input, // NCHW format
    const vector<vector<vector<vector<float>>>>& weights, // OIHW format
    const vector<float>& bias,
    int stride_h, int stride_w, int M) {

    int N = input.size();    // Batch size
    int C = input[0].size(); // Input Channels
    int H = input[0][0].size(); // Input Height
    int W = input[0][0][0].size(); // Input Width

    int K = weights.size();  // Output Channels
    int kernel_h = weights[0][0].size(); // Kernel Height
    int kernel_w = weights[0][0][0].size(); // Kernel Width

    int out_h = (H - kernel_h) / stride_h + 1;
    int out_w = (W - kernel_w) / stride_w + 1;

    vector<vector<vector<vector<float>>>> output(N, vector<vector<vector<float>>>(K, vector<vector<float>>(out_h, vector<float>(out_w))));

    // KN2Col transformation with M kernels
    vector<vector<float>> kn2col(out_h * out_w, vector<float>(C * kernel_h * kernel_w));
    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            int row_idx = oh * out_w + ow;
            for (int c = 0; c < C; ++c) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int h_idx = oh * stride_h + kh;
                        int w_idx = ow * stride_w + kw;
                        int col_idx = c * kernel_h * kernel_w + kh * kernel_w + kw;
                        kn2col[row_idx][col_idx] = input[0][c][h_idx][w_idx];
                    }
                }
            }
        }
    }

    // Perform GEMM (General Matrix Multiplication) with M kernels
    for (int k = 0; k < K; k += M) {
        for (int m = 0; m < M && (k + m) < K; ++m) {
            for (int i = 0; i < out_h * out_w; ++i) {
                float sum = bias[k + m];
                for (int j = 0; j < C * kernel_h * kernel_w; ++j) {
                    sum += weights[k + m][j / (kernel_h * kernel_w)][(j / kernel_w) % kernel_h][j % kernel_w] * kn2col[i][j];
                }
                output[0][k + m][i / out_w][i % out_w] = sum;
            }
        }
    }
    return output;
}

int main() {
    vector<vector<vector<vector<float>>>> input = {{{{1, 3, 2, 1}, {4, 6, 5, 2}, {7, 9, 8, 3}, {5, 2, 1, 4}}}};
    vector<vector<vector<vector<float>>>> weights = {{{{1, 0}, {0, -1}}}, {{{0, 1}, {-1, 0}}}}; // Example with multiple kernels
    vector<float> bias = {0, 0};
    int stride_h = 1, stride_w = 1;
    int M = 2; // Number of kernels processed at once
    
    auto output = gemm_convolution(input, weights, bias, stride_h, stride_w, M);
    
    for (const auto& n : output) {
        for (const auto& k : n) {
            for (const auto& row : k) {
                for (float val : row) {
                    cout << val << " ";
                }
                cout << endl;
            }
        }
    }
    return 0;
}


*/