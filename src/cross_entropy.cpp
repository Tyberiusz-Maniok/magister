#include "cross_entropy.h"
#include "consts.h"
#include <cmath>

using namespace lamp;

Tensor& CrossEntorpyLoss::forward(Tensor& x) {
    return x;
}

Tensor& CrossEntorpyLoss::sanity_check(Tensor& x) {
    return forward(x);
}

Tensor& CrossEntorpyLoss::backward(Tensor& grad, float lr) {
    return grad;
}

Tensor& CrossEntorpyLoss::loss(Tensor& pred, Tensor& target) {
    float* loss = (float*) mkl_calloc(pred.shape->n, sizeof(float), MALLOC_ALIGN);

    #pragma omp parallel for simd collapse(2)
    for (int n = 0; n < pred.shape->n; n++) {
        for (int w = 0; w < pred.shape->w; w++) {
            *(loss+n) -= target[n] * std::log(pred.at(n, 1, 1, w));
        }
    }

    Tensor* out = new Tensor(loss, new Shape(pred.shape->n, 1, 1, 1));
    return *out;
}
