#pragma once
#include "layer.h"

namespace lamp {

class Model : Layer {
    public:
        Layer* net;

        Model(Layer* net);
        ~Model();

        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        Tensor& backward(Tensor& grad, float lr) override;
        void set_train(bool train) override;
};

}
