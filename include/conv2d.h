#pragma once
#include "layer.h"
#include "rng.h"
#include "activations.h"

namespace lamp {

class Conv2d : public Layer {
    public:
        Tensor* filters;
        Tensor* bias;
        int stride;
        int kernel;
        int in_c;
        int out_c;
        int out_h;
        int out_w;
        Activation& activation_fn;
        Tensor* input_col;

        Conv2d(int input, int output, int kernel, int stride, RandomGen& rng, Activation& activation_fn=identity);
        ~Conv2d();

        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        Tensor& backward(Tensor& grad, float lr) override;

        Tensor& im2col(Tensor& x);
        Tensor& col2im(Tensor& x, Shape& shape);
        void init_bias(Shape* shape, RandomGen& rng);
};

}
