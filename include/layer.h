#pragma once
#include "tensor.h"

namespace lamp {

class Layer {
    public:
    //     Tensor& weights;

        Layer(){};
        ~Layer(){};

        Tensor& forward(const Tensor& x);
        void backward();
};

}
