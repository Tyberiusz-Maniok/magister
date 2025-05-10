#pragma once
#include "layer.h"

namespace lamp {

class Flatten : Layer {
    public:
        int c;
        int h;
        int w;

        Flatten();

        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        Tensor& backward(Tensor& grad, float lr) override;
};

}