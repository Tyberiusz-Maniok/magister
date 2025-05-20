#pragma once
#include "layer.h"
#include "rng.h"
#include "activations.h"

namespace lamp {

class Linear : public Layer {
    public:
        // int input;
        // int output;
        Tensor* weights;
        Tensor* bias;
        Activation& activation_fn;

        Linear(int input, int output, Activation& activation_fn=identity);
        ~Linear();

        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        Tensor& backward(Tensor& grad, float lr) override;
};

}
