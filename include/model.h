#pragma once
#include "layer.h"
#include "cross_entropy.h"
#include "data_loader.h"

namespace lamp {

class Model : public Layer {
    public:
        Layer* net;
        float lr;
        StatTracker* stat_tracker;
        CrossEntorpyLoss* loss = new CrossEntorpyLoss();

        Model(Layer* net, float lr, StatTracker* stat_tracker);
        ~Model();

        TensorP forward(TensorP x) override;
        TensorP sanity_check(TensorP x) override;
        TensorP backward(TensorP grad, float lr) override;
        void set_train(bool train) override;

        void fit(DataLoader& data_loader);
};

}
