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
        Tensor* input;
        std::string name;
        StatTracker* stat_tracker;

        Layer();
        ~Layer();

        virtual Tensor& forward(Tensor& x)=0;
        virtual Tensor& sanity_check(Tensor& x)=0;
        virtual Tensor& backward(Tensor& grad, float lr)=0;

        virtual void set_train(bool train);
        virtual Tensor& forward_t(Tensor& x);
        virtual Tensor& backward_t(Tensor& grad, float lr);
};

}
