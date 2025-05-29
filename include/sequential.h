#pragma once
#include "layer.h"
#include <vector>

namespace lamp {

class Sequential : public Layer {
    public:
        std::vector<LayerP> layers;
        int layer_n;

        Sequential(std::vector<LayerP> layers, int layer_n);
        ~Sequential();

        TensorP forward(TensorP x) override;
        TensorP sanity_check(TensorP x) override;
        TensorP backward(TensorP grad, float lr) override;
        void set_train(bool train) override;
        void set_stat_tracker(StatTrackerP stat_tracker) override;

        TensorP forward_t(TensorP x) override;
        TensorP backward_t(TensorP grad, float lr);
};

typedef std::shared_ptr<Sequential> SequentialP;

}
