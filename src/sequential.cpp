#include "sequential.h"
#include <omp.h>

using namespace lamp;

Sequential::Sequential(Layer** layers, int layer_n) : layers(layers), layer_n(layer_n) {}

Tensor& Sequential::forward(Tensor& x) {
    Tensor* out;
    for (int i = 0; i < layer_n; i++) {
        out = &((*(layers+i))->forward(x));
    }
    return *out;
}

Tensor& Sequential::sanity_check(Tensor& x) {
    Tensor* out;
    for (int i = 0; i < layer_n; i++) {
        out = &((*(layers+i))->sanity_check(x));
    }
    return *out;
}

Tensor& Sequential::backward(Tensor& grad, float lr) {
    Tensor* dgrad;
    for (int i = layer_n - 1; i >= 0; i--) {
        dgrad = &((*(layers+i))->backward(grad, lr));
    }
    return *dgrad;
}

void Sequential::set_train(bool train) {
    this->train = train;
    for (int i = 0; i < layer_n; i++) {
        (*(layers+i))->train = train;
    }
}

Tensor& Sequential::forward_t(Tensor& x) {
    double start, end, time = 0;
    Tensor* out;
    for (int i = 0; i < layer_n; i++) {
        double start = omp_get_wtime();
        out = &((*(layers+i))->forward(x));
        double end = omp_get_wtime();
    }
    time += end - start;
    return *out;
}

Tensor& Sequential::backward_t(Tensor& grad, float lr) {
    double start, end, time = 0;
    Tensor* out;
    for (int i = 0; i < layer_n; i++) {
        double start = omp_get_wtime();
        out = &((*(layers+i))->backward(grad, lr));
        double end = omp_get_wtime();
    }
    time += end - start;
    return *out;
}
