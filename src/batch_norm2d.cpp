#include "batch_norm2d.h"

using namespace lamp;

BatchNorm2d::BatchNorm2d(float epsilon) : epsilon(epsilon) {}

Tensor& BatchNorm2d::forward(Tensor& x) {
    return x;
}

Tensor& BatchNorm2d::sanity_check(Tensor& x) {
    return forward(x);
}

Tensor& BatchNorm2d::backward(Tensor& grad) {
    return grad;
}
