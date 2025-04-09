#include "maxpool.h"
#include "consts.h"
#include <mkl.h>

using namespace lamp;

MaxPool::MaxPool(int kernel) : kernel(kernel), stride(kernel) {}

MaxPool::MaxPool(int kernel, int stride) : kernel(kernel), stride(stride) {}

Tensor& MaxPool::forward(Tensor& x) {
    int out_h = (x.shape->h - kernel) / stride + 1;
    int out_w = (x.shape->w - kernel) / stride + 1;

    float* data = (float*) mkl_malloc(x.shape->n * x.shape->c * out_h * out_w * sizeof(float), MALLOC_ALIGN);
    Tensor* out = new Tensor(data, new Shape(x.shape->n, x.shape->c, out_h, out_w));

    # pragma omp parallel for simd collapse(4)
    for (int n = 0; n < x.shape->n; n++) {
        for (int c = 0; c < x.shape->c; c++) {
            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    float max = x.at(n, c, h * stride, w * stride);
                    for (int kh = 0; kh < kernel; kh++) {
                        for (int kw = 0; kw < kernel; kw++) {
                            max = std::max(max, x.at(n, c, h * stride + kh, w * stride +kw));
                        }
                    }
                    *(data+out->flat_index(n, c, h, w)) = max;
                }
            }
        }
    }

    return *out; 
}

Tensor& MaxPool::sanity_check(Tensor& x) {
    return forward(x);
}

Tensor& MaxPool::backward(Tensor& grad) {
    return grad;   
}
