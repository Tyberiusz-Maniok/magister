#include "batch_norm2d.h"
#include "consts.h"
#include <cmath>

using namespace lamp;

BatchNorm2d::BatchNorm2d(float epsilon, float mul, float bias) : epsilon(epsilon), mul(mul), bias(bias) {}

BatchNorm2d::~BatchNorm2d() {}

Tensor& BatchNorm2d::forward(Tensor& x) {

    if (train) {
        this->input = &x;
    }

    Tensor* out = new Tensor(x);

    float* avgs = (float*) mkl_malloc(x.shape->n * x.shape->c * sizeof(float), MALLOC_ALIGN);
    float* stds = (float*) mkl_malloc(x.shape->n * x.shape->c * sizeof(float), MALLOC_ALIGN);

    #pragma omp parallel for
    for (int n = 0; n < x.shape->n; n++) {
        for (int c = 0; c < x.shape->c; c++) {
            float avg = out->avg2d(n, c);
            float std = std::sqrt(out->variance_from_avg2d(n, c, avg) + epsilon);
            for (int h = 0; h < x.shape->h; h++) {
                for (int w = 0; w < x.shape->w; w++) {
                    *(out->data + out->flat_index(n, c, h, w)) = (out->at(n, c, h, w) - avg) / std * mul + bias;
                }
            }
        }
    }

    return *out;
}

Tensor& BatchNorm2d::sanity_check(Tensor& x) {
    return forward(x);
}

Tensor& BatchNorm2d::backward(Tensor& grad, float lr) {

    float delta_b = grad.sum();
    bias -= delta_b * lr;
    // float delta_gamma = normalized * out;

    // float delta_out = grad * mul;
    // float delta_x = 1.0 / input->size * inv_var * (grad->shape->n - delta_out.sum()) - (delta_out * out).sum();

    return grad;
}
