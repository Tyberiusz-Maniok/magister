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

    Tensor* t1 = Tensor::random(new Shape(2,3,4,5));

    // t1->print();

    Conv2d* conv2d = new Conv2d(3, 6, 3, 1);

    Tensor& out = conv2d->sanity_check(*t1);
    // out.print();

    Tensor& grad = conv2d->backward(out, 0.1);

    // Tensor& col = conv2d->im2col(*t1);
    // col.print();
    // col.print_shape();


    // delete &col;
    delete &grad;
    // delete &out;
    delete conv2d;

    // Tensor* t1 = Tensor::random(new Shape(2, 1, 1, 5));
    // Linear* lin = new Linear(5,4);

    // Tensor& out = lin->forward(*t1);
    // out.print_shape();
    // Tensor& back = lin->backward(out, 0.1);
    // // back.print();

    // delete &back;
    // delete &out;
    // delete lin;

    return 0;
}
