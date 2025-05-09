#pragma once
#include "tensor.h"
#include <string>

namespace lamp {

struct DataBatch {
    Tensor* x;
    Tensor* y;

    DataBatch(Tensor* x, Tensor* y) : x(x), y(y) {}
};

class DataLoader {
    public:
        int total_size;
        int remaining_size;
        int total_batches;
        int current_batch;

        int batch_size;

        DataLoader(int batch_size);

        DataBatch* next_batch();
        bool has_next();

        static void read_img(std::string filename, float* mem_ptr);
};

}

