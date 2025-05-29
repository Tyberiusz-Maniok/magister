#include "model.h"

using namespace lamp;

Model::Model(Layer* net, float lr, StatTrackerP stat_tracker) : net(net), lr(lr) {
    this->stat_tracker = stat_tracker;
    this->net->set_stat_tracker(stat_tracker);
};

Model::~Model() {
    delete this->net;
    delete this->loss;
}

TensorP Model::forward(TensorP x) {
    return net->forward(x);
}

TensorP Model::sanity_check(TensorP x) {
    return net->sanity_check(x);
}

TensorP Model::backward(TensorP grad, float lr) {
    return net->backward(grad, lr);
}

TensorP Model::forward_t(TensorP x) {
    return net->forward_t(x);
}

TensorP Model::backward_t(TensorP grad, float lr) {
    return net->backward_t(grad, lr);
}

void Model::set_train(bool train) {
    this->train = train;
    net->set_train(train);
}

void Model::fit(DataLoaderP data_loader) {
    sanity_check(data_loader->next_batch()->x);
    while (data_loader.has_next()) {
        DataBatchP batch = data_loader->next_batch();
        TensorP pred = forward(batch->x);
        TensorP cr_loss = loss->loss(pred, batch->y);
        backward(cr_loss, lr);
    }
}
