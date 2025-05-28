#pragma once
#include "layer.h"
#include "rng.h"
#include "activations.h"

namespace lamp {

class Conv2d : public Layer {
    public:
        TensorP filters;
        TensorP bias;
        int stride;
        int kernel;
        int in_c;
        int out_c;
        int out_h;
        int out_w;
        Activation& activation_fn;
        TensorP input_col;

        Conv2d(int input, int output, int kernel, int stride, Activation& activation_fn=identity);
        ~Conv2d();

        TensorP forward(TensorP x) override;
        TensorP sanity_check(TensorP x) override;
        TensorP backward(TensorP grad, float lr) override;

        TensorP im2col(TensorP x);
        TensorP col2im(TensorP x, Shape* shape);
        void init_bias(Shape* shape);
};

}
