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

    DataBatch* db = dl->next_batch();
    Tensor* t1 = db->x;
    // t1->print_shape();

    // Tensor& out = ((Sequential*)model->net)->layers->forward(*t1);
    Tensor& out = model->sanity_check(*t1);

    // Conv2d* conv = new Conv2d(1,6,6,1);
    // Conv2d* conv2 = new Conv2d(6, 6, 6, 1);

    // Sequential* s1 = new Sequential({conv, conv2}, 1);
    // Tensor& out = conv->sanity_check(*t1);
    // Tensor& out = s1->sanity_check(*t1);
    // out.print_shape();

    Linear* lin = new Linear(4,5);
    delete lin;

    delete dl;
    delete &out;
    // delete t1;
    // delete s1;
    // delete conv;
    // delete conv2;
    delete model;
    delete db->y;
    delete db;

    return 0;
}
