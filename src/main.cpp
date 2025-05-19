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

    Tensor* t1 = Tensor::random(new Shape(2, 3, 5, 4));

    // t1->print();

    Conv2d* conv2d = new Conv2d(3, 6, 3, 1);

    Tensor& out = conv2d->sanity_check(*t1);
    out.print();

    Tensor& col = conv2d->im2col(*t1);
    // col.print();
    // col.print_shape();


    delete &col;
    delete &out;
    delete conv2d;

    return 0;
}
