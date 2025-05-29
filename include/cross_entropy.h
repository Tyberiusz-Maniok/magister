#pragma once
#include "layer.h"

namespace lamp {

class CrossEntorpyLoss : public Layer {
    public:
        TensorP forward(TensorP x) override;
        TensorP sanity_check(TensorP x) override;
        TensorP backward(TensorP grad, float lr) override;

        TensorP loss(TensorP pred, TensorP target);
};

typedef std::shared_ptr<CrossEntorpyLoss> CrossEntorpyLossP;

}
