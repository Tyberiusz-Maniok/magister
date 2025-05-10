#include "flatten.h"

using namespace lamp;

Flatten::Flatten() {
    this->input = nullptr;
}

Tensor& Flatten::forward(Tensor& x) {
    if (train) {
        this->c = x.shape->c;
        this->h = x.shape->h;
        this->w = x.shape->w;
    }
    x.reshape(x.shape->n, 1, 1, x.shape->c * x.shape->h * x.shape->w);
    return x;
}
Tensor& Flatten::sanity_check(Tensor& x) {
    return forward(x);
}

Tensor& Flatten::backward(Tensor& grad, float lr) {
    grad.reshape(grad.shape->n, c, h, w);
    return grad;
}
