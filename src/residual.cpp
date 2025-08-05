#include "residual.h"

using namespace lamp;

Residual::Residual(LayerP block) : block(block) {}

TensorP Residual::forward(TensorP x) {
    return block->forward(x)->add(x);
}

TensorP Residual::sanity_check(TensorP x) {
    return forward(x);
}

TensorP Residual::backward(TensorP grad, float lr) {
    TensorP block_grad = block->backward(grad, lr);
    return grad->add(block_grad);
}
