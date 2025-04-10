#include <omp.h>
#include <cstdio>
#include <mkl.h>
#include "rng.h"
#include "tensor.h"
#include "conv2d.h"

using namespace lamp;

int main() {

    RandomGen* rng = new RandomGen(123);

    Tensor* t1 = Tensor::random(new Shape(1, 1, 4, 5), *rng);
    // Tensor* t2 = Tensor::random(new Shape(1,1, 5, 7), *rng);

    // Tensor& t3 = t1->matmul(*t2);

    // t3.print();

    Conv2d* conv = new Conv2d(1, 3, 3, 1, *rng);

    // conv->filters->print();

    Tensor& out = conv->sanity_check(*t1);

    Tensor& col = conv->im2col(*t1);

    Tensor& grad = conv->backward(out);
    // conv->filters->print();

    // t1->print();
    // out.print();
    grad.print();
    // col.print();

    // Tensor& im = conv->col2im(col, *(t1->shape));

    // im.print();

    delete &grad;
    // delete &im;
    delete &col;
    delete &out;
    delete conv;
    // delete &t3;
    // delete t2;
    // delete t1;
    delete rng;

    return 0;
}
