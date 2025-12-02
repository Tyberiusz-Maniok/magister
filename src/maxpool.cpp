#include "maxpool.h"
#include "consts.h"
#include <cstring>
#include <omp.h>

using namespace lamp;

MaxPool::MaxPool(int kernel) : kernel(kernel), stride(kernel) {}

MaxPool::MaxPool(int kernel, int stride) : kernel(kernel), stride(stride) {}

MaxPool::~MaxPool() {
    free(this->max_indices);
}

TensorP MaxPool::forward(TensorP x) {
    if (train) {
        this->input = x;
    }

    float* data = (float*) aligned_alloc(MALLOC_ALIGN, x->shape->n * x->shape->c * out_h * out_w * sizeof(float));
    TensorP out = std::shared_ptr<Tensor>(new Tensor(data, new Shape(x->shape->n, x->shape->c, out_h, out_w)));

    // if (train) {
        // std::memset(this->max_indices, 0, x->shape->n * x->shape->c * out_h * out_w * sizeof(int));
        // this->max_indices = (int*) mkl_realloc(this->max_indices, x->shape->n * x->shape->c * out_h * out_w * sizeof(int));
    // }

    // Extract values for GPU kernel (NCHW format)
    int x_n = x->shape->n;
    int x_c = x->shape->c;
    int x_stride_n = x->strides->n;
    int x_stride_c = x->strides->c;
    int x_stride_h = x->strides->h;
    int x_stride_w = x->strides->w;
    
    int out_stride_n = out->strides->n;
    int out_stride_c = out->strides->c;
    int out_stride_h = out->strides->h;
    int out_stride_w = out->strides->w;
    
    float* x_dev = x->d_data;
    float* out_dev = out->d_data;
    
    // Allocate max_indices on device if training
    int device_num = omp_get_default_device();
    int* indices_dev = nullptr;
    if (train) {
        indices_dev = (int*) omp_target_alloc(x_n * x_c * out_h * out_w * sizeof(int), device_num);
    }
    
    if (train) {
        #pragma omp target teams distribute parallel for collapse(4) is_device_ptr(x_dev, out_dev, indices_dev)
        for (int n = 0; n < x_n; n++) {
            for (int c = 0; c < x_c; c++) {
                for (int h = 0; h < out_h; h++) {
                    for (int w = 0; w < out_w; w++) {
                        int idx = n * x_stride_n + c * x_stride_c + h * stride * x_stride_h + w * stride * x_stride_w;
                        float max = *(x_dev+idx);
                        int out_idx = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w;
                        *(indices_dev+out_idx) = idx;
                        
                        for (int kh = 0; kh < kernel; kh++) {
                            for (int kw = 0; kw < kernel; kw++) {
                                idx = n * x_stride_n + c * x_stride_c + (h * stride + kh) * x_stride_h + (w * stride + kw) * x_stride_w;
                                float val = *(x_dev+idx);
                                if (val > max) {
                                    max = val;
                                    *(indices_dev+out_idx) = idx;
                                }
                            }
                        }
                        *(out_dev+out_idx) = max;
                    }
                }
            }
        }
        
        // Copy indices back to host
        int host_num = omp_get_initial_device();
        omp_target_memcpy(max_indices, indices_dev, x_n * x_c * out_h * out_w * sizeof(int), 0, 0, host_num, device_num);
        omp_target_free(indices_dev, device_num);
    } else {
        #pragma omp target teams distribute parallel for collapse(4) is_device_ptr(x_dev, out_dev)
        for (int n = 0; n < x_n; n++) {
            for (int c = 0; c < x_c; c++) {
                for (int h = 0; h < out_h; h++) {
                    for (int w = 0; w < out_w; w++) {
                        int idx = n * x_stride_n + c * x_stride_c + h * stride * x_stride_h + w * stride * x_stride_w;
                        float max = *(x_dev+idx);
                        int out_idx = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w;
                        
                        for (int kh = 0; kh < kernel; kh++) {
                            for (int kw = 0; kw < kernel; kw++) {
                                idx = n * x_stride_n + c * x_stride_c + (h * stride + kh) * x_stride_h + (w * stride + kw) * x_stride_w;
                                float val = *(x_dev+idx);
                                if (val > max) {
                                    max = val;
                                }
                            }
                        }
                        *(out_dev+out_idx) = max;
                    }
                }
            }
        }
    }

    return out; 
}

TensorP MaxPool::sanity_check(TensorP x) {
    this->out_h = (x->shape->h - kernel) / stride + 1;
    this->out_w = (x->shape->w - kernel) / stride + 1;
    this->max_indices = (int*) aligned_alloc(MALLOC_ALIGN, x->shape->n * x->shape->c * out_h * out_w * sizeof(int));
    return forward(x);
}

TensorP MaxPool::backward(TensorP grad, float lr) {
    float* input_data = (float*) aligned_alloc(MALLOC_ALIGN, input->size * sizeof(float));
    memset(input_data, 0, input->size * sizeof(float));
    TensorP input_grad = TensorP(new Tensor(input_data, new Shape(*(input->shape))));

    int num_indices = input->shape->n * input->shape->c * out_h * out_w;
    float* grad_dev = input_grad->d_data;
    
    // Allocate indices on device
    int device_num = omp_get_default_device();
    int* indices_dev = (int*) omp_target_alloc(num_indices * sizeof(int), device_num);
    int host_num = omp_get_initial_device();
    omp_target_memcpy(indices_dev, max_indices, num_indices * sizeof(int), 0, 0, device_num, host_num);
    
    #pragma omp target teams distribute parallel for is_device_ptr(indices_dev, grad_dev)
    for (int i = 0; i < num_indices; i++) {
        *(grad_dev + *(indices_dev+i)) = 1;
    }
    
    omp_target_free(indices_dev, device_num);

    return input_grad;   
}
