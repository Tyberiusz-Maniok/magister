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
        int in_c;
        int out_c;
        int out_h;
        int out_w;
        activ_fn activation_fn;

        Conv2d(int input, int output, int kernel, int stride, RandomGen& rng, activ_fn activation_fn=Layer::identity);
        ~Conv2d();

        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        Tensor& backward(Tensor& grad) override;

        Tensor& im2col(Tensor& x);
        Tensor& col2im(Tensor& x, Shape& shape);
};

}
