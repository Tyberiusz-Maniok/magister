#include "layer.h"

using namespace lamp;

Layer::Layer() {}

Layer::~Layer() {
    if (this->input != nullptr) {
        delete input;
    }
}

void Layer::set_train(bool train) {
    this->train = train;
}

void Layer::set_stat_tracker(StatTracker* stat_tracker) {
    this->stat_tracker = stat_tracker;
}

Tensor& Layer::forward_t(Tensor& x) {
    double start_time = omp_get_wtime();
    Tensor& out = forward(x);
    double end_time = omp_get_wtime();
    stat_tracker->add(Stats(this->name, end_time - start_time));
    return out;
}

Tensor& Layer::backward_t(Tensor& grad, float lr) {
    double start_time = omp_get_wtime();
    Tensor& out = backward(grad, lr);
    double end_time = omp_get_wtime();
    stat_tracker->add(Stats(this->name + "_back", end_time - start_time));
    return out;
}
