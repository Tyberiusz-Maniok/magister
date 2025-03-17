#include "linear.h"

Linear::Linear(int input, int, output, RanomGen& rng, activ_fn activation_fn) : Layer(), activation_fn(activation_fn) {
    int size = input * output;
    float* data = mkl_malloc(size * sizeof(float), 32);
    int* shape = mkl_malloc(2 * sizeof(int), 32);

    rng.populate(data);
    this->weights = new Tensor(size, data, shape, 2);
}

Linear::~Linear() {
    delete this->weights;
    delete this->bias;
}