#include "maxpool.h"
#include "consts.h"
#include "mkl.h"

using namespace lamp;

MaxPool::MaxPool(int kernel) : kernel(kernel), stride(kernel) {}

MaxPool::MaxPool(int kernel, int stride) : kernel(kernel), stride(stride) {}

MaxPool::~MaxPool() {
    mkl_free(this->max_indices);
}

TensorP MaxPool::forward(TensorP x) {
    if (train) {
        input = x;
    }

    float* data = (float*) mkl_malloc(x->shape->n * x->shape->c * out_h * out_w * sizeof(float), MALLOC_ALIGN);
    TensorP out = std::shared_ptr<Tensor>(new Tensor(data, new Shape(x->shape->n, x->shape->c, out_h, out_w)));

    if (train) {
        this->max_indices = (int*) mkl_malloc(x->shape->n * x->shape->c * out_h * out_w * sizeof(int), MALLOC_ALIGN);
    }

    # pragma omp parallel for simd collapse(4)
    for (int n = 0; n < x->shape->n; n++) {
        for (int c = 0; c < x->shape->c; c++) {
            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    int idx = x->flat_index(n, c, h * stride, w * stride);
                    float max = x->data[idx];
                    if (train) {
                        *(max_indices+out->flat_index(n, c, h, w)) = idx;
                    }
                    for (int kh = 0; kh < kernel; kh++) {
                        for (int kw = 0; kw < kernel; kw++) {
                            idx = x->flat_index(n, c, h * stride + kh, w * stride +kw);
                            float val = x->data[idx];
                            if (val > max) {
                                max = val;
                                if (train) {
                                    *(max_indices+out->flat_index(n, c, h, w)) = idx;
                                }
                            }
                        }
                    }
                    *(data+out->flat_index(n, c, h, w)) = max;
                }
            }
        }
    }

    return out; 
}

TensorP MaxPool::sanity_check(TensorP x) {
    this->out_h = (x->shape->h - kernel) / stride + 1;
    this->out_w = (x->shape->w - kernel) / stride + 1;
    return forward(x);
}

TensorP MaxPool::backward(TensorP grad, float lr) {
    float* input_data = (float*) mkl_calloc(input->size, sizeof(float), MALLOC_ALIGN);
    TensorP input_grad = TensorP(new Tensor(input_data, new Shape(*(input->shape))));

    #pragma omp parallel for simd
    for (int i = 0; i < input->shape->n * input->shape->c * out_h * out_w; i++) {
        *(input_data + *(max_indices + i)) = 1;
    }

    return input_grad;   
}
