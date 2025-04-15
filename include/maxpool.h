#pragma once
#include "layer.h"

namespace lamp {

class MaxPool : Layer {
    public:
        int kernel;
        int stride;
        int out_h;
        int out_w;
        int* max_indices;

        MaxPool(int kernel);
        MaxPool(int kernel, int stride);
        
        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        Tensor& backward(Tensor& grad, float lr) override;
};

}