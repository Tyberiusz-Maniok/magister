#include "batch_norm2d.h"
#include "consts.h"
#include <cmath>

using namespace lamp;

BatchNorm2d::BatchNorm2d(float epsilon, float mul, float bias) : epsilon(epsilon), mul(mul), bias(bias) {}

BatchNorm2d::~BatchNorm2d() {}

TensorP BatchNorm2d::forward(TensorP x) {

    if (train) {
        this->input = x;
    }

    float* avgs = (float*) mkl_malloc(x->shape->n * x->shape->c * sizeof(float), MALLOC_ALIGN);
    float* stds = (float*) mkl_malloc(x->shape->n * x->shape->c * sizeof(float), MALLOC_ALIGN);

    float* out = (float*) mkl_malloc(x->size * sizeof(float), MALLOC_ALIGN);

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < x->shape->n; n++) {
        for (int c = 0; c < x->shape->c; c++) {
            float avg = x->avg2d(n, c);
            float std = std::sqrt(x->variance_from_avg2d(n, c, avg) + epsilon);
            for (int h = 0; h < x->shape->h; h++) {
                for (int w = 0; w < x->shape->w; w++) {
                    *(out + x->flat_index(n, c, h, w)) = (x->at(n, c, h, w) - avg) / std * mul + bias;
                }
            }
        }
    }

    return TensorP(new Tensor(out, new Shape(*(x->shape))));
}

TensorP BatchNorm2d::sanity_check(TensorP x) {
    return forward(x);
}

TensorP BatchNorm2d::backward(TensorP grad, float lr) {

    float delta_b = grad->sum();
    bias -= delta_b * lr;
    // float delta_gamma = normalized * out;

    // float delta_out = grad * mul;
    // float delta_x = 1.0 / input->size * inv_var * (grad->shape->n - delta_out.sum()) - (delta_out * out).sum();

    return grad;
}
