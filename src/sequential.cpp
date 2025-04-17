#include "sequential.h"

using namespace lamp;

Sequential::Sequential(Layer** layers, int layer_n) : layers(layers), layer_n(layer_n) {}

Tensor& Sequential::forward(Tensor& x) {
    Tensor* out;
    for (int i = 0; i < layer_n; i++) {
        out = &((*(layers+i))->forward(x));
    }
    return *out;
}

Tensor& Sequential::sanity_check(Tensor& x) {
    Tensor* out;
    for (int i = 0; i < layer_n; i++) {
        out = &((*(layers+i))->sanity_check(x));
    }
    return *out;
}

Tensor& Sequential::backward(Tensor& grad, float lr) {
    Tensor* dgrad;
    for (int i = layer_n - 1; i >= 0; i--) {
        dgrad = &((*(layers+i))->backward(grad, lr));
    }
    return *dgrad;
}
