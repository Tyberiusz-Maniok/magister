#include "layer.h"

using namespace lamp;

Layer::Layer() {}

Layer::~Layer() {
}

void Layer::set_train(bool train) {
    this->train = train;
}

void Layer::set_stat_tracker(StatTrackerP stat_tracker) {
    this->stat_tracker = stat_tracker;
}

TensorP Layer::forward_t(TensorP x) {
    double start_time = omp_get_wtime();
    TensorP out = forward(x);
    double end_time = omp_get_wtime();
    stat_tracker->add(Stats(this->name, end_time - start_time));
    return out;
}

TensorP Layer::backward_t(TensorP grad, float lr) {
    double start_time = omp_get_wtime();
    TensorP out = backward(grad, lr);
    double end_time = omp_get_wtime();
    stat_tracker->add(Stats(this->name + "_back", end_time - start_time));
    return out;
}
