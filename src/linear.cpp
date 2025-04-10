#include "linear.h"
#include <mkl.h>
#include <stdexcept>
#include "consts.h"

using namespace lamp;

Linear::Linear(int input, int output, RandomGen& rng, activ_fn activation_fn) : activation_fn(activation_fn) {
    Shape* shape = new Shape(1, 1, output, input);

    this->weights = Tensor::random(shape, rng);
    this->bias = Tensor::random(new Shape(*shape), rng);
}

Linear::~Linear() {
    delete this->weights;
    delete this->bias;
}

Tensor& Linear::forward(Tensor& x) {
    return x.matmul(*weights, bias);
}

Tensor& Linear::sanity_check(Tensor& x) {
    if (x.shape->w != this->weights->shape->h) {
        throw std::runtime_error("invalid shape");
    }
    return forward(x);
}

Tensor& Linear::backward(Tensor& grad) {
    // float* delta = (float*) mkl_malloc(weights->shape->h * weights->shape->w * sizeof(float), MALLOC_ALIGN);

    //TODO apply relu derivative to grad
    Tensor& delta_w = input->matmul(grad, nullptr, CblasTrans, CblasNoTrans);
    Tensor& input_grad = grad.matmul(*weights, nullptr, CblasNoTrans, CblasTrans);

    *weights -= delta_w; // TODO *lr
    *bias -= grad;

    return input_grad;
}
