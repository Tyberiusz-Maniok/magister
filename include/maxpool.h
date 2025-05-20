#pragma once
#include "layer.h"

namespace lamp {

class MaxPool : public Layer {
    public:
        int kernel;
        int stride;
        int out_h;
        int out_w;
        int* max_indices;

        MaxPool(int kernel);
        MaxPool(int kernel, int stride);
        ~MaxPool();
        
        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        Tensor& backward(Tensor& grad, float lr) override;
};

}