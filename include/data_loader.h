#pragma once
#include "tensor.h"

namespace lamp {

class DataLoader {
    public:
        int current;
        int total;

        int batch_size;

        Tensor& next_batch();
};

}

