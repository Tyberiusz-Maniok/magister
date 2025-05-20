#include "sequential.h"
#include <omp.h>

using namespace lamp;

Sequential::Sequential(std::vector<Layer*> layers, int layer_n) : layers(layers), layer_n(layer_n) {}

Sequential::~Sequential() {
    for (int i = 0; i < layer_n; i++) {
        delete layers[i];
    }
}

Tensor& Sequential::forward(Tensor& x) {
    Tensor* out = new Tensor(x);
    for (int i = 0; i < layer_n; i++) {
        out = &(layers[i]->forward(*out));
    }
    return *out;
}

Tensor& Sequential::sanity_check(Tensor& x) {
    Tensor* out = new Tensor(x);
    for (int i = 0; i < layer_n; i++) {
        out = &(layers[i]->sanity_check(*out));
    }
    return *out;
}

Tensor& Sequential::backward(Tensor& grad, float lr) {
    Tensor* dgrad = new Tensor(grad);
    for (int i = layer_n - 1; i >= 0; i--) {
        dgrad = &(layers[i]->backward(*dgrad, lr));
    }
    return *dgrad;
}

void Sequential::set_train(bool train) {
    this->train = train;
    for (int i = 0; i < layer_n; i++) {
        layers[i]->set_train(train);
    }
}

void Sequential::set_stat_tracker(StatTracker* stat_tracker) {
    this->stat_tracker = stat_tracker;
    for (int i = 0; i < layer_n; i++) {
        layers[i]->set_stat_tracker(stat_tracker);
    }
}

Tensor& Sequential::forward_t(Tensor& x) {
    double start, end, time = 0;
    Tensor* out = new Tensor(x);
    for (int i = 0; i < layer_n; i++) {
        double start = omp_get_wtime();
        out = &(layers[i]->forward_t(*out));
        double end = omp_get_wtime();
    }
    time += end - start;
    return *out;
}

Tensor& Sequential::backward_t(Tensor& grad, float lr) {
    double start, end, time = 0;
    Tensor* out = new Tensor(grad);
    for (int i = 0; i < layer_n; i++) {
        double start = omp_get_wtime();
        out = &(layers[i]->backward_t(*out, lr));
        double end = omp_get_wtime();
    }
    time += end - start;
    return *out;
}
