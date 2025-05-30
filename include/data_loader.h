#pragma once
#include "tensor.h"
#include <string>

namespace lamp {

struct DataBatch {
    TensorP x;
    TensorP y;

    DataBatch(TensorP x, TensorP y) : x(x), y(y) {}
    // ~DataBatch() {
    //     delete x;
    //     delete y;
    // }
};

typedef std::shared_ptr<DataBatch> DataBatchP;

class DataLoader {
    public:
        int total_size;
        int remaining_size;
        int total_batches;
        int current_batch;

        int batch_size;

        DataLoader(int batch_size);

        DataBatchP next_batch();
        bool has_next();
        void reset_epoch();

        static void read_img(std::string filename, float* mem_ptr);
};

typedef std::shared_ptr<DataLoader> DataLoaderP;

}

