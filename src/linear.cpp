#include "linear.h"
#include <mkl.h>

using namespace lamp;

Linear::Linear(int input, int output, RandomGen& rng, activ_fn activation_fn) : activation_fn(activation_fn) {
    int size = input * output;
    // float* data = (float*) mkl_malloc(size * sizeof(float), 32);
    int* shape = (int*) mkl_malloc(2 * sizeof(int), 32);
    *(shape) = input;
    *(shape+1) = output;

    this->weights = &Tensor::random(shape, 2, rng);
}

Linear::~Linear() {
    delete this->weights;
    delete this->bias;
}