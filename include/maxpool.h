#pragma once
#include "layer.h"

namespace lamp {

class MaxPool : Layer {
    public:
        int kernel;
        int stride;

        MaxPool(int kernel);
        MaxPool(int kernel, int stride);
        
        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        void backward() override;
};

}