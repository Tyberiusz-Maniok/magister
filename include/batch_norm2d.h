#pragma once
#include "layer.h"

namespace lamp {

class BatchNorm2d : public Layer {
    public:
        float epsilon;
        float mul;
        float bias;

        // Tensor* avgs;
        // Tensor* vars;
        // Tensor* xhat;

        BatchNorm2d(float epsilon, float mul = 0.9, float bias = 0.1);
        ~BatchNorm2d();

        TensorP forward(TensorP x) override;
        TensorP sanity_check(TensorP x) override;
        TensorP backward(TensorP grad, float lr) override;
};

typedef std::shared_ptr<BatchNorm2d> BatchNorm2dP;

}
