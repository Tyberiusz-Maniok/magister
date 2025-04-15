#pragma once
#include "layer.h"

namespace lamp {

class BatchNorm2d : Layer {
    public:
        float epsilon;
        float mul = 0.9;
        float bias = 0.1;

        BatchNorm2d(float epsilon);

        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        Tensor& backward(Tensor& grad, float lr) override;
};

}
