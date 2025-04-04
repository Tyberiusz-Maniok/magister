#include "layer.h"

using namespace lamp;

void Layer::identity(Tensor& x) {}

void Layer::relu(Tensor& x) {
    #pragma omp parallel for
    for (int i = 0; i < x.size; i++) {
        if (*(x.data+i) < 0) {
            *(x.data+i) = 0;
        }
    }
}