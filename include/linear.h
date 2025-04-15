#pragma once
#include "layer.h"
#include "rng.h"

namespace lamp {

class Linear : Layer {
    public:
        // int input;
        // int output;
        Tensor* weights;
        Tensor* bias;
        activ_fn activation_fn;

        Linear(int input, int output, RandomGen& rng, activ_fn activation_fn=Layer::identity);
        ~Linear();

        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        Tensor& backward(Tensor& grad, float lr) override;
};

}
