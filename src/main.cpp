#include <omp.h>
#include <cstdio>
#include "mkl.h"
#include "rng.h"
#include "tensor.h"
#include "conv2d.h"
#include "linear.h"
#include "stats.h"
#include "data_loader.h"
#include <string>
#include "models.h"

using namespace lamp;

int main() {

    printf("Starting...\n");
    #pragma omp parallel master
    {
        printf("%i\n", omp_get_num_threads());
        // printf("%i\n", omp_get_num_devices());
        // omp_set_default_device(2);
        // printf("%i\n", omp_get_num_devices());
        // printf("%i\n", omp_get_device_num());
    }

    DataLoaderP dl = DataLoaderP(new DataLoader(16));

    Model* m = models::vgg16();
    // Model* m = models::resnet18();
    // TensorP x = dl->next_batch()->x;

    // x->print_shape();
    // TensorP out = m->sanity_check(x);
    // out->print_shape();

    m->fit(dl);

    m->stat_tracker->to_csv("../results/resnet_t1_dyn.csv");


    // DataBatchP db = dl->next_batch();
    // Conv2dP conv = Conv2dP(new Conv2d(1,3,3,1, relu));
    // TensorP out = conv->sanity_check(db->x);
    // TensorP grad = conv->backward(out, 0.1);

    delete m;
    // TensorP out1 = m->forward_t(x);
    // out->print();

    return 0;
}
