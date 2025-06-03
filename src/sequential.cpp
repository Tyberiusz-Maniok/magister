#include "sequential.h"
#include <omp.h>

using namespace lamp;

Sequential::Sequential(std::vector<LayerP> layers, int layer_n) : layers(layers), layer_n(layer_n) {}

Sequential::~Sequential() {
    // for (int i = 0; i < layers.size(); i++) {
    //     delete layers[i];
    // }
}

TensorP Sequential::forward(TensorP x) {
    TensorP out = x;
    for (int i = 0; i < layers.size(); i++) {
        out = layers[i]->forward(out);
    }
    return out;
}

TensorP Sequential::sanity_check(TensorP x) {
    TensorP out = x;
    for (int i = 0; i < layers.size(); i++) {
        out = layers[i]->sanity_check(out);
    }
    return out;
}

TensorP Sequential::backward(TensorP grad, float lr) {
    TensorP dgrad = grad;
    for (int i = layers.size() - 1; i >= 0; i--) {
        dgrad = layers[i]->backward(dgrad, lr);
    }
    return dgrad;
}

void Sequential::set_train(bool train) {
    this->train = train;
    for (int i = 0; i < layers.size(); i++) {
        layers[i]->set_train(train);
    }
}

void Sequential::set_stat_tracker(StatTrackerP stat_tracker) {
    this->stat_tracker = stat_tracker;
    for (int i = 0; i < layers.size(); i++) {
        layers[i]->set_stat_tracker(stat_tracker);
    }
}

TensorP Sequential::forward_t(TensorP x) {
    double start, end, time = 0;
    TensorP out = x;
    for (int i = 0; i < layers.size(); i++) {
        double start = omp_get_wtime();
        out = layers[i]->forward_t(out);
        double end = omp_get_wtime();
    }
    time += end - start;
    return out;
}

TensorP Sequential::backward_t(TensorP grad, float lr) {
    double start, end, time = 0;
    TensorP out = grad;
    for (int i = layers.size() - 1; i >= 0; i--) {
        double start = omp_get_wtime();
        out = layers[i]->backward_t(out, lr);
        double end = omp_get_wtime();
    }
    time += end - start;
    return out;
}
