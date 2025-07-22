#pragma once
#include "layer.h"

namespace lamp {

class Residual : public Layer {
    public:
        LayerP block;

        Residual(LayerP block);

        TensorP forward(TensorP x) override;
        TensorP sanity_check(TensorP x) override;
        TensorP backward(TensorP grad, float lr) override;
};

}