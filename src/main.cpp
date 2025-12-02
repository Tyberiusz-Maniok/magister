#include <omp.h>
#include <cstdio>
#include "cuda_nvhpc_compat.h"
#include <cuda_runtime.h>
#include "rng.h"
#include "tensor.h"
#include "conv2d.h"
#include "linear.h"
#include "stats.h"
#include "data_loader.h"
#include <string>
#include "models.h"
#include "consts.h"

using namespace lamp;

int main() {

    printf("Starting...\n");
    fflush(stdout);
    
    // CRITICAL: Initialize CUDA context early
    omp_set_default_device(0);
    cudaSetDevice(0);
    cudaFree(0);  // Force CUDA context initialization
    
    DataLoaderP dl = DataLoaderP(new DataLoader(4));  // Batch size 4 for 4GB GPU
    printf("DataLoader created\n");
    fflush(stdout);

    Model* m = models::vgg16();
    printf("Model created\n");
    fflush(stdout);
    // Model* m = models::resnet18();
    // TensorP x = dl->next_batch()->x;

    // x->print_shape();
    // TensorP out = m->sanity_check(x);
    // out->print_shape();

    printf("Starting model fit...\n");
    fflush(stdout);
    m->fit(dl);
    printf("Model fit completed\n");
    fflush(stdout);

    m->stat_tracker->to_csv("../results/all_omp.csv");
    printf("Stats saved\n");
    fflush(stdout);


    // DataBatchP db = dl->next_batch();
    // Conv2dP conv = Conv2dP(new Conv2d(1,3,3,1, relu));
    // TensorP out = conv->sanity_check(db->x);
    // TensorP grad = conv->backward(out, 0.1);

    delete m;
    // TensorP out1 = m->forward_t(x);
    // out->print();

    return 0;
}
