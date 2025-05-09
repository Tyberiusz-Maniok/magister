#include "data_loader.h"
#include "consts.h"
#include <iostream>
#include <fstream>
#include <mkl.h>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace lamp;

DataLoader::DataLoader(int batch_size) : batch_size(batch_size) {
    this->total_size = 60000; //TODO move to const
    this->remaining_size = this->total_size;
}

DataBatch* DataLoader::next_batch() {
    int batch_s = batch_size;
    if (remaining_size < batch_size) {
        batch_s = remaining_size;
    }
    float* result = (float*) mkl_malloc(batch_s * IMAGE_SIZE * sizeof(float), MALLOC_ALIGN);
    #pragma omp parallel for
    for (int i = 0; i < batch_s; i++) {
        read_img("../data/mod/1.jpg", result + i * IMAGE_SIZE);
    }
    Tensor* x = new Tensor(result, new Shape(batch_s, 1, IMAGE_H, IMAGE_W));
    return new DataBatch(x, nullptr);
}

bool DataLoader::has_next() {
    return true;
}

void DataLoader::read_img(std::string filename, float* mem_ptr) {
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    img.convertTo(img, CV_32F, 1.0 / 255, 0);
    std::memcpy(mem_ptr, img.data, img.size().height * img.size().width * sizeof(float));
    img.release();
}
