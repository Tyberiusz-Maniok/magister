#include <omp.h>
#include <cstdio>
#include <mkl.h>
#include "rng.h"
#include "tensor.h"
#include "conv2d.h"
#include "stats.h"
#include "data_loader.h"

using namespace lamp;

int main() {

    // RandomGen* rng = new RandomGen(123);
    // StatTracker* stats = new StatTracker();

    // Tensor* t1 = Tensor::random(new Shape(2, 3, 4, 5), *rng);
    // // Tensor* t2 = Tensor::random(new Shape(1,1, 5, 7), *rng);

    // // Tensor& t3 = t1->matmul(*t2);

    // // t3.print();

    // Conv2d* conv = new Conv2d(3, 6, 3, 1, *rng);
    // conv->name = "conv1";
    // conv->stat_tracker = stats;

    // // conv->filters->print();

    // Tensor& out = conv->sanity_check(*t1);
    // Tensor& out1 = conv->forward_t(*t1);

    // Tensor& col = conv->im2col(*t1);

    // Tensor& grad = conv->backward(out1, 0.1);
    // // conv->filters->print();

    // // t1->print();
    // // out1.print();
    // // col.print();
    // // conv->filters->print();
    // grad.print();


    // // Tensor& im = conv->col2im(col, *(t1->shape));

    // // im.print();

    // delete &grad;
    // // delete &im;
    // delete &col;
    // delete &out;
    // delete &out1;
    // delete conv;
    // // delete &t3;
    // // delete t2;
    // // delete t1;
    // delete stats;
    // delete rng;

    // DataLoader* dl = new DataLoader(1);

    // Tensor& img = dl->next_batch();

    // img.print();

    // delete &img;
    // delete dl;

    return 0;
}
