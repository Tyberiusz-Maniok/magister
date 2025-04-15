#pragma once
#include "layer.h"

namespace lamp {

class Leaf {
    public:
        Layer* layer;
        Leaf* dependencies;
        int n_dep;
        Tensor* grad;

    Leaf(Layer* layer, Leaf* dependencies, int n_dep);

    Tensor& backward(Tensor& grad, float lr);
};

}
