#include "data_loader.h"
#include "consts.h"
#include <iostream>
#include <fstream>
#include "mkl.h"
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace lamp;

DataLoader::DataLoader(int batch_size) : batch_size(batch_size) {
    this->total_size = DATA_SIZE;
    this->remaining_size = this->total_size;
}

DataBatchP DataLoader::next_batch() {
    int batch_s = batch_size;
    if (remaining_size < batch_size) {
        batch_s = remaining_size;
    }
    float* data_x = (float*) mkl_malloc(batch_s * IMAGE_SIZE * sizeof(float), MALLOC_ALIGN);
    float* data_y = (float*) mkl_calloc(batch_s * CLASSES, sizeof(float), MALLOC_ALIGN);

    std::ifstream y_file(CLASS_FILE);
    std::string buffer;

    #pragma omp parallel for
    for (int i = 0; i < batch_s; i++) {
        read_img("../data/mod/1.jpg", data_x + i * IMAGE_SIZE);
        std::getline(y_file, buffer);
        *(data_y + i * CLASSES + std::stoi(buffer)) = 1.0;
    }
    y_file.close();
    remaining_size -= batch_s;

    TensorP x = std::shared_ptr<Tensor>(new Tensor(data_x, new Shape(batch_s, 1, IMAGE_H, IMAGE_W)));
    TensorP y = std::shared_ptr<Tensor>(new Tensor(data_y, new Shape(batch_s, 1, 1, CLASSES)));
    return std::shared_ptr<DataBatch>(new DataBatch(x, y));
}

bool DataLoader::has_next() {
    return remaining_size > 0;
}

void DataLoader::reset_epoch() {
    this->remaining_size = this->total_size;
}

void DataLoader::read_img(std::string filename, float* mem_ptr) {
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    img.convertTo(img, CV_32F, 1.0 / 255, 0);
    std::memcpy(mem_ptr, img.data, img.size().height * img.size().width * sizeof(float));
    img.release();
}
