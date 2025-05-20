#pragma once
#include "layer.h"

namespace lamp {

class CrossEntorpyLoss : public Layer {
    public:
        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        Tensor& backward(Tensor& grad, float lr) override;

        Tensor& loss(Tensor& pred, Tensor& target);
};

}
