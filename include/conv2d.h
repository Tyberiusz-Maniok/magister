#pragma once
#include "layer.h"
#include "tensor.h"
#include "rng.h"

namespace lamp {

class Conv2d : Layer {
    public:
        Tensor* filters;
        int stride;
        int k;
        activ_fn activation_fn;

        Conv2d(int input, int output, int kernel, RandomGen& rng, activ_fn activation_fn=identity);
        ~Conv2d();

        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        void backward() override;

        Tensor& im2col(Tensor& x);
        Tensor& col2im(Tensor& x);
};

}
