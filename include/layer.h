#pragma once
#include "tensor.h"

namespace lamp {

class Layer {
    public:
    //     Tensor& weights;

        Layer(){};
        ~Layer(){};

        virtual Tensor& forward(Tensor& x)=0;
        virtual Tensor& sanity_check(Tensor& x)=0;
        virtual void backward()=0;
};

}
