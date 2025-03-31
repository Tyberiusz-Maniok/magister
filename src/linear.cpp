#include "linear.h"
#include <mkl.h>
#include <stdexcept>

using namespace lamp;

Linear::Linear(int input, int output, RandomGen& rng, activ_fn activation_fn) : activation_fn(activation_fn) {
    Shape* shape = new Shape(1, 1, input, output);

    this->weights = Tensor::random(shape, rng);
    this->bias = Tensor::random(shape, rng);
}

Linear::~Linear() {
    delete this->weights;
    delete this->bias;
}

Tensor& Linear::forward(Tensor& x) {
    Tensor& out = x.matmul(*weights);
    out += *(this->bias);
    return out;
}

Tensor& Linear::sanity_check(Tensor& x) {
    if (x.shape->w != this->weights->shape->h) {
        throw std::runtime_error("invalid shape");
    }
    return forward(x);
}

void Linear::backward() {
    
}
