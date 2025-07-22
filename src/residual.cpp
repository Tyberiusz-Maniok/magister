#include "residual.h"

using namespace lamp;

Residual::Residual(LayerP block) : block(block) {}

TensorP Residual::forward(TensorP x) {
    // return block->forward(x) + x.get();
    return block->forward(x);
}

TensorP Residual::sanity_check(TensorP x) {
    return forward(x);
}

TensorP Residual::backward(TensorP grad, float lr) {
    return grad;
}
