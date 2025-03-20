#pragma once
#include "layer.h"

namespace lamp {

class Conv2d : Layer {
    public:
        Tensor* filters;
        activ_fn activation_fn;

        Conv2d();
        ~Conv2d();

        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        void backward() override;
};

}
