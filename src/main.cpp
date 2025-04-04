#include <omp.h>
#include <cstdio>
#include <mkl.h>
#include "rng.h"
#include "tensor.h"
#include "conv2d.h"

using namespace lamp;

int main() {

    RandomGen* rng = new RandomGen(123);

    Tensor* t1 = Tensor::random(new Shape(1, 1, 4, 4), *rng);

    Conv2d* conv = new Conv2d(1, 3, 3, *rng);

    Tensor& out = conv->sanity_check(*t1);

    delete &out;
    delete conv;
    delete t1;
    delete rng;

    return 0;
}
