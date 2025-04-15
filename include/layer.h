#pragma once
#include "tensor.h"
#include <functional>

namespace lamp {

typedef std::function<void(Tensor&)> activ_fn;

class Layer {
    public:
        bool train = true;
        Tensor* input;

        Layer();
        ~Layer();

        virtual Tensor& forward(Tensor& x)=0;
        virtual Tensor& sanity_check(Tensor& x)=0;
        virtual Tensor& backward(Tensor& grad, float lr)=0;

        static void identity(Tensor& x);
        static void relu(Tensor& x);
};

}
