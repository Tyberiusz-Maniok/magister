#pragma once
#include "layer.h"
#include "cross_entropy.h"

namespace lamp {

class Model : Layer {
    public:
        Layer* net;
        CrossEntorpyLoss* loss = new CrossEntorpyLoss();

        Model(Layer* net);
        ~Model();

        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        Tensor& backward(Tensor& grad, float lr) override;
        void set_train(bool train) override;
};

}
