#pragma once
#include "layer.h"
#include "tensor.h"
#include <functional>
#include "rng.h"

typedef std::function<void(Tensor&)> activ_fn;

class Linear : Layer {
    public:
        Tensor* weights;
        Tensor* bias;
        activ_fn activation_fn;

        // Linear(std::function<void(const Tensor&)> mt_func=identity);
        Linear(int input, int output, RandomGen& rng, activation_fn=identity);
        ~Linear();

        void identity(Tensor& x) {};

        Tensor& forward(const Tensor& x);
};