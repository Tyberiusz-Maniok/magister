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

    Model* model = models::lenet();

    // ((Conv2d*) ((Sequential*) model->net )->layers)->filters->print();

    DataLoader* dl = new DataLoader(1);

    Tensor* t1 = dl->next_batch()->x;
    t1->print_shape();

    // Tensor& out = ((Sequential*)model->net)->layers->forward(*t1);
    Tensor& out = model->sanity_check(*t1);
    out.print();

    delete dl;
    delete &out;
    // delete t1;
    delete model;

    return 0;
}
