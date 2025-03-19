#pragma once
#include "layer.h"
#include "tensor.h"
#include <functional>
#include "rng.h"

namespace lamp {

typedef std::function<void(Tensor&)> activ_fn;

class Linear : Layer {
    public:
        Tensor* weights;
        Tensor* bias;
        activ_fn activation_fn;

        Linear(int input, int output, RandomGen& rng, activ_fn activation_fn=[](Tensor& x){});
        ~Linear();

        Tensor& forward(Tensor& x) override;
        Tensor& sanity_check(Tensor& x) override;
        void backward() override;

        void identity(Tensor& x) {};
};

}
