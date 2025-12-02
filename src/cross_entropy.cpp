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

    int batch_n = pred->shape->n;
    int num_classes = pred->shape->w;
    float* pred_dev = pred->d_data;
    float* target_dev = target->d_data;
    int pred_stride_n = pred->strides->n;
    int pred_stride_c = pred->strides->c;
    int pred_stride_h = pred->strides->h;
    int pred_stride_w = pred->strides->w;
    
    // Allocate loss on device
    int device_num = omp_get_default_device();
    float* loss_dev = (float*) omp_target_alloc(batch_n * sizeof(float), device_num);
    int host_num = omp_get_initial_device();
    omp_target_memcpy(loss_dev, loss, batch_n * sizeof(float), 0, 0, device_num, host_num);
    
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(pred_dev, target_dev, loss_dev)
    for (int n = 0; n < batch_n; n++) {
        for (int w = 0; w < num_classes; w++) {
            int pred_idx = n * pred_stride_n + 1 * pred_stride_c + 1 * pred_stride_h + w * pred_stride_w;
            #pragma omp atomic
            *(loss_dev+n) -= *(target_dev+n) * log(*(pred_dev+pred_idx));
        }
    }
    
    // Copy result back
    omp_target_memcpy(loss, loss_dev, batch_n * sizeof(float), 0, 0, host_num, device_num);
    omp_target_free(loss_dev, device_num);

    return TensorP(new Tensor(loss, new Shape(pred->shape->n, 1, 1, 1)));
}
