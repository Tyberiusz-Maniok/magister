#pragma once
#include "layer"

namespace lamp {

class MaxPool : Layer {
    public:
        int kernel;

        MaxPool(int kernel);
        
        virtual Tensor& forward(Tensor& x)=0;
        virtual Tensor& sanity_check(Tensor& x)=0;
        virtual void backward()=0;
};

}