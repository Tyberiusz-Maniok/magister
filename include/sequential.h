#pragma once
#include "layer.h"
#include <vector>

namespace lamp {

class Sequential : public Layer {
    public:
        std::vector<Layer*> layers;
        int layer_n;

        Sequential(std::vector<Layer*> layers, int layer_n);
        ~Sequential();

        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        Tensor& backward(Tensor& grad, float lr) override;
        void set_train(bool train) override;
        void set_stat_tracker(StatTracker* stat_tracker) override;

        Tensor& forward_t(Tensor& x) override;
        Tensor& backward_t(Tensor& grad, float lr);
};

}
