#include "model.h"

using namespace lamp;

Model::Model(Layer* net, float lr) : net(net), lr(lr) {};

Model::~Model() {
    delete this->net;
    delete this->loss;
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
    while (data_loader.has_next()) {
        DataBatch* batch = data_loader.next_batch();
        Tensor& pred = forward(*(batch->x));
        Tensor& cr_loss = loss->loss(pred, *(batch->y));
        backward(cr_loss, lr);
    }
}
