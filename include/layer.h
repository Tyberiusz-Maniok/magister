#pragma once
#include "tensor.h"
#include <functional>
#include <omp.h>
#include "stats.h"
#include "activations.h"

namespace lamp {

class Layer {
    public:
        bool train = true;
        TensorP input = nullptr;
        std::string name;
        StatTracker* stat_tracker;

        Layer();
        virtual ~Layer();

        virtual TensorP forward(TensorP x)=0;
        virtual TensorP sanity_check(TensorP x)=0;
        virtual TensorP backward(TensorP grad, float lr)=0;

        virtual void set_train(bool train);
        virtual void set_stat_tracker(StatTracker* stat_tracker);
        virtual TensorP forward_t(TensorP x);
        virtual TensorP backward_t(TensorP grad, float lr);
};

}
