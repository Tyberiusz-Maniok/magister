#include "linear.h"
#include "mkl.h"
#include <stdexcept>
#include "consts.h"

using namespace lamp;

Linear::Linear(int input, int output, Activation& activation_fn) : activation_fn(activation_fn) {
    this->weights = Tensor::random(new Shape(1, 1, input, output));
    this->bias = Tensor::random(new Shape(1, 1, 1, output));
}

Linear::~Linear() {
    delete this->weights;
    delete this->bias;
}

Tensor& Linear::forward(Tensor& x) {
    if (train) {
        this->input = &x;
    }

    // assume x is flattened
    x.reshape(1, 1, x.shape->n, x.shape->w);
    Tensor& out = x.matmul(*weights, bias);
    out.reshape(out.shape->h, 1, 1, out.shape->w);
    activation_fn.forward(out);
    return out;
}

Tensor& Linear::sanity_check(Tensor& x) {
    if (x.shape->w != weights->shape->h) {
        throw std::runtime_error("invalid shape");
    }
    return forward(x);
}

Tensor& Linear::backward(Tensor& grad, float lr) {
    activation_fn.backward(grad);
    Tensor& delta_w = input->matmul(grad, nullptr, CblasTrans, CblasNoTrans);
    Tensor& input_grad = grad.matmul(*weights, nullptr, CblasNoTrans, CblasTrans);

    weights->mulsub(delta_w, lr);
    bias->mulsub(grad, lr);

    return input_grad;
}
