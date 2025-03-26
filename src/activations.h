#include "activations.h"
#include <omp.h>

using namespace lamp;

void identity(Tensor& x) {}

void relu(Tensor& x) {
    #pragma omp parallel for
    for (int i = 0; i < x.size; i++) {
        if (*(x.data+i) < 0) {
            *(x.data+i) = 0;
        }
    }
}
