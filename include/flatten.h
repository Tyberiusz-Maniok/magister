#pragma once
#include "layer.h"

namespace lamp {

class Flatten : public Layer {
    public:
        int c;
        int h;
        int w;

        Flatten();

        TensorP forward(TensorP x) override;
        TensorP sanity_check(TensorP x) override;
        TensorP backward(TensorP grad, float lr) override;
};

typedef std::shared_ptr<Flatten> FlattenP;

}