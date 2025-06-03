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

    DataLoaderP dl = DataLoaderP(new DataLoader(2));

    Model* m = models::vgg16();
    // TensorP x = dl->next_batch()->x;

    // x->print_shape();
    // TensorP out = m->sanity_check(x);
    // out->print_shape();

    m->fit(dl);

    delete m;
    // TensorP out1 = m->forward_t(x);
    // out->print();

    return 0;
}
