#pragma once
#include "tensor.h"

namespace lamp {

class DataLoader {
    public:
        int total_size;
        int total_batches;
        int current_batch;

        int batch_size;

        DataLoader(int batch_size);

        Tensor& next_batch();
};

}

