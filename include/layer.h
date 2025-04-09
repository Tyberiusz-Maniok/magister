#pragma once
#include "tensor.h"

namespace lamp {

typedef std::function<void(Tensor&)> activ_fn;

class Layer {
    public:
        Layer(){};
        ~Layer(){};

        virtual Tensor& forward(Tensor& x)=0;
        virtual Tensor& sanity_check(Tensor& x)=0;
        virtual Tensor& backward(Tensor& grad)=0;

        static void identity(Tensor& x);
        static void relu(Tensor& x);
};

}
