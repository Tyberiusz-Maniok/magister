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
    // delete this->weights;
    // delete this->bias;
}

TensorP Linear::forward(TensorP x) {
    if (train) {
        this->input = x;
    }

    // assume x is flattened
    x->reshape(1, 1, x->shape->n, x->shape->w);
    TensorP out = x->matmul(weights, bias);
    out->reshape(out->shape->h, 1, 1, out->shape->w);
    activation_fn.forward(out);
    return out;
}

TensorP Linear::sanity_check(TensorP x) {
    // if (x->shape->w != weights->shape->h) {
    //     throw std::runtime_error("invalid shape");
    // }
    return forward(x);
}

TensorP Linear::backward(TensorP grad, float lr) {
    activation_fn.backward(grad);
    TensorP agrad = grad->avg_grad();
    grad->reshape(1,1, grad->shape->n * grad->shape->c * grad->shape->h, grad->shape->w);
    TensorP delta_w = input->matmul(grad, nullptr, CblasTrans, CblasNoTrans);
    TensorP input_grad = grad->matmul(weights, nullptr, CblasNoTrans, CblasTrans);

    weights->mulsub(delta_w, lr);
    bias->mulsub(agrad, lr);

    // delete &grad;
    // delete &delta_w;
    // delete &agrad;

    return input_grad;
}
