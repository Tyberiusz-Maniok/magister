#include "cross_entropy.h"
#include "consts.h"
#include <cmath>
#include <cstring>

using namespace lamp;

TensorP CrossEntorpyLoss::forward(TensorP x) {
    return x;
}

TensorP CrossEntorpyLoss::sanity_check(TensorP x) {
    return forward(x);
}

TensorP CrossEntorpyLoss::backward(TensorP grad, float lr) {
    return grad;
}

TensorP CrossEntorpyLoss::loss(TensorP pred, TensorP target) {
    float* loss = (float*) aligned_alloc(MALLOC_ALIGN, pred->shape->n * sizeof(float));
    memset(loss, 0, pred->shape->n * sizeof(float));

    #pragma omp parallel for simd collapse(2)
    for (int n = 0; n < pred->shape->n; n++) {
        for (int w = 0; w < pred->shape->w; w++) {
            *(loss+n) -= target->data[n] * std::log(pred->at(n, 1, 1, w));
        }
    }

    return TensorP(new Tensor(loss, new Shape(pred->shape->n, 1, 1, 1)));
}
