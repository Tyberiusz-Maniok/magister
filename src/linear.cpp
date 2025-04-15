#include "linear.h"
#include <mkl.h>
#include <stdexcept>
#include "consts.h"

using namespace lamp;

Linear::Linear(int input, int output, RandomGen& rng, activ_fn activation_fn) : activation_fn(activation_fn) {
    Shape* shape = new Shape(1, 1, input, output);

    this->weights = Tensor::random(shape, rng);
    this->bias = Tensor::random(new Shape(1, 1, 1, output), rng);
}

Linear::~Linear() {
    delete this->weights;
    delete this->bias;
}

Tensor& Linear::forward(Tensor& x) {
    if (train) {
        this->input = &x;
    }

    return x.matmul(*weights, bias);
}

Tensor& Linear::sanity_check(Tensor& x) {
    if (x.shape->w != weights->shape->h) {
        throw std::runtime_error("invalid shape");
    }
    return forward(x);
}

Tensor& Linear::backward(Tensor& grad, float lr) {
    //TODO apply relu derivative to grad
    Tensor& delta_w = input->matmul(grad, nullptr, CblasTrans, CblasNoTrans);
    Tensor& input_grad = grad.matmul(*weights, nullptr, CblasNoTrans, CblasTrans);

    weights->mulsub(delta_w, lr);
    bias->mulsub(grad, lr);

    return input_grad;
}
