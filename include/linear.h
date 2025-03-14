#pragma once
#include "layer.h"
#include <functional>
#define noop

// typedef void (*mt_func)(Tensor&);
typedef std::function<void(Tensor&)> mt_func;

class Linear : Layer {
    public:
        Tensor& weights;
        Tensor& bias;
        mt_func activation_fn;

        // Linear(std::function<void(const Tensor&)> mt_func=identity);
        Linear(mt_func activation_fn=identity);
        ~Linear();

        void identity(Tensor& x) {};
};