#include "activations.h"

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
    #pragma omp parallel for
    for (int i = 0; i < x->size; i++) {
        if (*(x->data+i) < 0) {
            *(x->data+i) = 0;
        }
    }
}

void Activation::f_relu_backward(TensorP x) {
    #pragma omp parallel for
    for (int i = 0; i < x->size; i++) {
        if (*(x->data+i) < 0) {
            *(x->data+i) = 0;
        } else {
            *(x->data+i) = 1;
        }
    }
}
