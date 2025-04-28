#include "model.h"

using namespace lamp;

Model::Model(Layer* net) : net(net) {};

Model::~Model() {
    delete this->net;
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
