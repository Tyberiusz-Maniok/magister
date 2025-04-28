#pragma once
#include "tensor.h"
#include <functional>
#include <omp.h>

namespace lamp {

typedef std::function<void(Tensor&)> activ_fn;

class Layer {
    public:
        bool train = true;
        Tensor* input;

        Layer();
        ~Layer();

        virtual Tensor& forward(Tensor& x)=0;
        virtual Tensor& sanity_check(Tensor& x)=0;
        virtual Tensor& backward(Tensor& grad, float lr)=0;

        virtual void set_train(bool train) {
            this->train = train;
        }

        virtual Tensor& forward_t(Tensor& x) {
            double start_time = omp_get_wtime();
            Tensor& out = forward(x);
            double end_time = omp_get_wtime();
            return out;
        }

        virtual Tensor& backward_t(Tensor& grad, float lr) {
            double start_time = omp_get_wtime();
            Tensor& out = backward(grad, lr);
            double end_time = omp_get_wtime();
            return out;
        }

        static void identity(Tensor& x);
        static void relu(Tensor& x);
};

}
