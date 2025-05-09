#pragma once
#include "layer.h"
#include "cross_entropy.h"
#include "data_loader.h"

namespace lamp {

class Model : Layer {
    public:
        Layer* net;
        float lr;
        CrossEntorpyLoss* loss = new CrossEntorpyLoss();

        Model(Layer* net, float lr);
        ~Model();

        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        Tensor& backward(Tensor& grad, float lr) override;
        void set_train(bool train) override;

        void fit(DataLoader& data_loader);
};

}
