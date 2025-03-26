#pragma once
#include "layer.h"
#include "tensor.h"
#include "rng.h"

namespace lamp {

class Linear : Layer {
    public:
        Tensor* weights;
        Tensor* bias;
        activ_fn activation_fn;

        Linear(int input, int output, RandomGen& rng, activ_fn activation_fn=identity);
        ~Linear();

        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        void backward() override;
};

}
