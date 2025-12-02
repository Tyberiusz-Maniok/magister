#include "activations.h"
#include <omp.h>

using namespace lamp;

Activation::Activation(activ_fn forward_fn, activ_fn backward_fn) : forward_fn(forward_fn), backward_fn(backward_fn) {}

void Activation::forward(TensorP x) {
    this->forward_fn(x);
}

void Activation::backward(TensorP x) {
    this->backward_fn(x);
}

void Activation::f_identity(TensorP x) {}

void Activation::f_relu(TensorP x) {
    float* dev_ptr = x->d_data;
    int data_size = x->size;
    #pragma omp target teams distribute parallel for is_device_ptr(dev_ptr)
    for (int i = 0; i < data_size; i++) {
        if (*(dev_ptr+i) < 0) {
            *(dev_ptr+i) = 0;
        }
    }
}

void Activation::f_relu_backward(TensorP x) {
    float* dev_ptr = x->d_data;
    int data_size = x->size;
    #pragma omp target teams distribute parallel for is_device_ptr(dev_ptr)
    for (int i = 0; i < data_size; i++) {
        if (*(dev_ptr+i) < 0) {
            *(dev_ptr+i) = 0;
        } else {
            *(dev_ptr+i) = 1;
        }
    }
}
