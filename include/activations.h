#pragma once
#include <functional>
#include "tensor.h"

namespace lamp {

// typedef std::function<void(Tensor&)> activ_fn;
typedef void(*activ_fn)(Tensor&);

class Activation {
    public:
        activ_fn forward_fn;
        activ_fn backward_fn;

        Activation(activ_fn forward_fn , activ_fn backward_fn);

        void forward(Tensor& x);
        void backward(Tensor& x);

        static void f_identity(Tensor& x);
        static void f_relu(Tensor& x);
        static void f_relu_backward(Tensor& x);
};

static Activation identity = Activation(Activation::f_identity, Activation::f_identity);
static Activation relu = Activation(Activation::f_relu, Activation::f_relu_backward);

}
