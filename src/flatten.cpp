#include "flatten.h"

using namespace lamp;

Flatten::Flatten() {
    this->input = nullptr;
}

TensorP Flatten::forward(TensorP x) {
    if (train) {
        this->c = x->shape->c;
        this->h = x->shape->h;
        this->w = x->shape->w;
    }
    x->reshape(x->shape->n, 1, 1, x->shape->c * x->shape->h * x->shape->w);
    return x;
}
TensorP Flatten::sanity_check(TensorP x) {
    return forward(x);
}

TensorP Flatten::backward(TensorP grad, float lr) {
    grad->reshape(grad->shape->n, c, h, w);
    return grad;
}

TensorP Flatten::forward_t(TensorP x) {
    return forward(x);
}

TensorP Flatten::backward_t(TensorP grad, float lr) {
    return backward(grad, lr);
}
