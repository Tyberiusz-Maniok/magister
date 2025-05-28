#pragma once
#include "layer.h"
#include "rng.h"
#include "activations.h"

namespace lamp {

class Linear : public Layer {
    public:
        // int input;
        // int output;
        TensorP weights;
        TensorP bias;
        Activation& activation_fn;

        Linear(int input, int output, Activation& activation_fn=identity);
        ~Linear();

        TensorP forward(TensorP x) override;
        TensorP sanity_check(TensorP x) override;
        TensorP backward(TensorP grad, float lr) override;
};

}
