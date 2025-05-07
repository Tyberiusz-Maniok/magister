#pragma once
#include "tensor.h"
#include <string>

namespace lamp {

class DataLoader {
    public:
        int total_size;
        int remaining_size;
        int total_batches;
        int current_batch;

        int batch_size;

        DataLoader(int batch_size);

        Tensor& next_batch();

        static void read_img(std::string filename, float* mem_ptr);
};

}

