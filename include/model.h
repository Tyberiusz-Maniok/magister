#pragma once
#include "layer.h"
#include "cross_entropy.h"
#include "data_loader.h"

namespace lamp {

class Model : public Layer {
    public:
        Layer* net;
        float lr;
        CrossEntorpyLoss* loss = new CrossEntorpyLoss();

        Model(Layer* net, float lr, StatTrackerP stat_tracker);
        ~Model();

        TensorP forward(TensorP x) override;
        TensorP sanity_check(TensorP x) override;
        TensorP backward(TensorP grad, float lr) override;
        TensorP forward_t(TensorP x) override;
        TensorP backward_t(TensorP grad, float lr) override;
        void set_train(bool train) override;

        void fit(DataLoaderP data_loader);
};

typedef std::shared_ptr<Model> ModelP;

}
