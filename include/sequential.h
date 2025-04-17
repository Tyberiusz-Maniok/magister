#pragma once
#include "layer.h"

namespace lamp {

class Sequential : Layer {
    public:
        Layer** layers;
        int layer_n;

        Sequential(Layer** layers, int layer_n);

        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        Tensor& backward(Tensor& grad, float lr) override;
};

}
