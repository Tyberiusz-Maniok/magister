#pragma once
#include "tensor.h"

class Layer {
    public:
    //     Tensor& weights;

        Layer();
        ~Layer();

        Tensor& forward(const Tensor& x);
        void backward();
};