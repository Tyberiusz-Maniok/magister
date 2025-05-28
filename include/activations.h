#pragma once
#include <functional>
#include "tensor.h"

namespace lamp {

// typedef std::function<void(TensorP)> activ_fn;
typedef void(*activ_fn)(TensorP);

class Activation {
    public:
        activ_fn forward_fn;
        activ_fn backward_fn;

        Activation(activ_fn forward_fn , activ_fn backward_fn);

        void forward(TensorP x);
        void backward(TensorP x);

        static void f_identity(TensorP x);
        static void f_relu(TensorP x);
        static void f_relu_backward(TensorP x);
};

static Activation identity = Activation(Activation::f_identity, Activation::f_identity);
static Activation relu = Activation(Activation::f_relu, Activation::f_relu_backward);

}
