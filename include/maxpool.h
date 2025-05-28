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
        
        TensorP forward(TensorP x) override;
        TensorP sanity_check(TensorP x) override;
        TensorP backward(TensorP grad, float lr) override;
};

}