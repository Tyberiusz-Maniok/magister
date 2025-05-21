#include "model.h"

using namespace lamp;

Model::Model(Layer* net, float lr, StatTracker* stat_tracker) : net(net), lr(lr), stat_tracker(stat_tracker) {
    this->net->set_stat_tracker(stat_tracker);
};

Model::~Model() {
    delete this->net;
    delete this->loss;
    delete this->stat_tracker;
}

Tensor& Model::forward(Tensor& x) {
    return net->forward(x);
}

Tensor& Model::sanity_check(Tensor& x) {
    return net->sanity_check(x);
}

Tensor& Model::backward(Tensor& grad, float lr) {
    return net->backward(grad, lr);
}

void Model::set_train(bool train) {
    this->train = train;
    net->set_train(train);
}

void Model::fit(DataLoader& data_loader) {
    sanity_check(*(data_loader.next_batch()->x));
    while (data_loader.has_next()) {
        DataBatch* batch = data_loader.next_batch();
        Tensor& pred = forward(*(batch->x));
        Tensor& cr_loss = loss->loss(pred, *(batch->y));
        backward(cr_loss, lr);
    }
}
