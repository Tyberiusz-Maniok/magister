#include "tensor.h"
#include <omp.h>
#include <stdlib.h>
#include "mkl.h"
#include "consts.h"
#include <cstring>

#include <cstdio>

using namespace lamp;

int accum_size(Shape* shape) {
    return shape->n * shape->c * shape->h * shape->w;
}

void Tensor::set_strides(Shape* shape) {
    this->strides = new Shape(
        shape->c * shape->h * shape->w,
        shape->h * shape->w,
        shape->w,
        1
    );
}

Tensor::Tensor(float* data, Shape* shape) : data(data), shape(shape) {
    this->size = accum_size(shape);
    set_strides(shape);
}

Tensor::Tensor(float* data, Shape* shape, int size) : data(data), shape(shape), size(size) {
    set_strides(shape);
}

Tensor::Tensor(Tensor& other) : size(other.size), shape(new Shape(*(other.shape))), strides(new Shape(*(other.strides))) {
    this->data = (float*) mkl_malloc(other.size * sizeof(float), MALLOC_ALIGN);
    std::memcpy(other.data, this->data, other.size * sizeof(float));
}

Tensor::Tensor(std::shared_ptr<Tensor> other) : size(other->size), shape(new Shape(*(other->shape))), strides(new Shape(*(other->strides))) {
    this->data = (float*) mkl_malloc(other->size * sizeof(float), MALLOC_ALIGN);
    std::memcpy(other->data, this->data, other->size * sizeof(float));
}

Tensor::~Tensor() {
    mkl_free(this->data);
    delete this->shape;
    delete this->strides;
}

Tensor& Tensor::operator=(Tensor& other) {
    return *(new Tensor(other));
}

Tensor& Tensor::operator+(Tensor& other) {
    Tensor& result = *(new Tensor(*this));
    result += other;
    return result;
}

Tensor& Tensor::operator-(Tensor& other) {
    Tensor& result = *(new Tensor(*this));
    result -= other;
    return result;
}

Tensor& Tensor::operator*(Tensor& other) {
    Tensor& result = *(new Tensor(*this));
    result *= other;
    return result;
}

Tensor& Tensor::operator/(Tensor& other) {
    Tensor& result = *(new Tensor(*this));
    result /= other;
    return result;
}

Tensor& Tensor::operator+=(Tensor& other) {
    #pragma omp parallel for simd
    for (int i=0; i < size; i++) {
        *(data+i) += *(other.data+i);
    }
    return *this;
}

Tensor& Tensor::operator-=(Tensor& other) {
    #pragma omp parallel for simd
    for (int i=0; i < size; i++) {
        *(data+i) -= *(other.data+i);
    }
    return *this;
}

Tensor& Tensor::operator*=(Tensor& other) {
    #pragma omp parallel for simd
    for (int i=0; i < size; i++) {
        *(data+i) *= *(other.data+i);
    }
    return *this;
}

Tensor& Tensor::operator/=(Tensor& other) {
    #pragma omp parallel for simd
    for (int i=0; i < size; i++) {
        *(data+i) /= *(other.data+i);
    }
    return *this;
}

Tensor& Tensor::operator*=(float other) {
    #pragma omp parallel for simd
    for (int i=0; i < this->size; i++) {
        *(data+i) *= other;
    }
    return *this;
}

void Tensor::mulsub(std::shared_ptr<Tensor> other, float mul) {
    #pragma omp parallel for simd
    for (int i = 0; i < size; i++) {
        *(data+i) -= *(other->data+i) * mul;
    }
}

int Tensor::flat_index(int n, int c, int h, int w) {
    return n * strides->n + c * strides->c + h * strides->h + w * strides->w;
}

float Tensor::at(int n, int c, int h, int w) {
    return *(data+flat_index(n, c, h, w));
}

float Tensor::operator[](int idx) {
    return *(data+idx);
}

float Tensor::dot(Tensor& other) {
    return cblas_sdot(size, data, 1, other.data, 1);
}

std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> other, std::shared_ptr<Tensor> bias, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb) {
    int m = this->shape->h;
    int k = this->shape->w;
    int n = other->shape->w;

    int lda = k;
    int ldb = n;
    int ldc = n;

    if (transa == CblasTrans) {
        m = k;
        k = this->shape->h;
        lda = m;
    }
    if (transb == CblasTrans) {
        n = other->shape->h;
        ldc = n;
        ldb = k;
    }

    float* result = (float*) mkl_malloc(m * n * sizeof(float), MALLOC_ALIGN);
    float beta = 0;
    if (bias != nullptr) {
        Tensor::bias_cpy(bias->data, result, bias->size, this->shape->n);
        beta = 1;
    }
    cblas_sgemm(CblasRowMajor, transa, transb, m, n, k, 1, this->data, lda, other->data, ldb, beta, result, ldc);

    return std::shared_ptr<Tensor>(new Tensor(result, new Shape(1, 1, m, n), m*n));
}

std::shared_ptr<Tensor> Tensor::batched_matmul(std::shared_ptr<Tensor> other, std::shared_ptr<Tensor> bias, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb) {
    int m = this->shape->h;
    int k = this->shape->w;
    int n = other->shape->w;

    int lda = k;
    int ldb = n;
    int ldc = n;

    if (transa == CblasTrans) {
        m = k;
        k = this->shape->h;
        lda = m;
    }
    if (transb == CblasTrans) {
        n = other->shape->h;
        ldc = n;
        ldb = k;
    }

    float* result = (float*) mkl_malloc(m * n * other->shape->n * sizeof(float), MALLOC_ALIGN);
    float beta = 0;
    if (bias != nullptr) {
        Tensor::bias_cpy(bias->data, result, bias->size, this->shape->n);
        beta = 1;
    }

    int in_stride = other->strides->n;
    int out_stride = m * n;
    #pragma omp parallel for
    for (int i = 0; i < other->shape->n; i++) {
        cblas_sgemm(CblasRowMajor, transa, transb, m, n, k, 1, this->data, lda, other->data+(in_stride*i), ldb, beta, result+(out_stride*i), ldc);
    }

    return std::shared_ptr<Tensor>(new Tensor(result, new Shape(other->shape->n, 1, m, n), m * n * other->shape->n));
}

std::shared_ptr<Tensor> Tensor::avg_grad() {
    if (shape->n == 1) {
        return std::shared_ptr<Tensor>(new Tensor(*this));
    }
    float* result = (float*) mkl_calloc(shape->c * shape->h * shape->w, sizeof(float), MALLOC_ALIGN);

    #pragma omp parallel for
    for (int i = 0; i < size / shape->n; i++) {
        for (int n = 0; n < shape->n; n++) {
            *(result+i) += *(data+i+(n*strides->n));
        }
        *(result+i) /= shape->n;
    }

    return std::shared_ptr<Tensor>(new Tensor(result, new Shape(1, shape->c, shape->h, shape->w)));
}

void Tensor::reshape(int n, int c, int h, int w) {
    this->shape->n = n;
    this->shape->c = c;
    this->shape->h = h;
    this->shape->w = w;
    this->strides->w = 1;
    this->strides->h = w;
    this->strides->c = h * w;
    this->strides->n = c * h * w;
}

float Tensor::sum() {
    float result = 0;
    #pragma omp parallel for simd reduction(+:result)
    for (int i = 0; i < size; i++) {
        result += *(data+i);
    }
    return result;
}

float Tensor::avg() {
    return sum() / size;
}

float Tensor::variance() {
    float result = 0;
    float mean = avg();
    #pragma omp parallel for simd reduction(+:result)
    for (int i = 0; i < size; i++) {
        float diff = *(data+i) - mean;
        result += diff * diff;
    }
    return result / size;
}

float Tensor::variance_from_avg(float avg) {
    float result = 0;
    #pragma omp parallel for simd reduction(+:result)
    for (int i = 0; i < size; i++) {
        result += *(data+i) - avg;
    }
    return result / size;
}

float Tensor::sum2d(int n, int c) {
    float result = 0;
    int start_idx = n * strides->n + c * strides->c;
    #pragma omp parallel for simd reduction(+:result)
    for (int i = start_idx; i < start_idx + shape->h * shape->w; i++) {
        result += *(data+i);
    }
    return result;
}


float Tensor::avg2d(int n, int c) {
    return sum2d(n, c) / (shape->h * shape->w);
}

float Tensor::variance2d(int n, int c) {
    float result = 0;
    float mean = avg2d(n, c);
    int start_idx = n * strides->n + c * strides->c;
    #pragma omp parallel for simd reduction(+:result)
    for (int i = start_idx; i < start_idx + shape->h * shape->w; i++) {
        float diff = *(this->data+i) - mean;
        result += diff * diff;
    }
    return result / size;
}

float Tensor::variance_from_avg2d(int n, int c, float avg) {
    float result = 0;
    int start_idx = n * strides->n + c * strides->c;
    #pragma omp parallel for simd reduction(+:result)
    for (int i = start_idx; i < start_idx + shape->h * shape->w; i++) {
        float diff = *(this->data+i) - avg;
        result += diff * diff;
    }
    return result / size;
}

void Tensor::print_shape() {
    printf("Shape: %i, %i, %i, %i\n", shape->n, shape->c, shape->h, shape->w);
}

void Tensor::print() {
    print_shape();
    for (int n = 0; n < shape->n; n++) {
        printf("[");
        for (int c = 0; c < shape->c; c++) {
            printf("[");
            for (int h = 0; h < shape->h; h++) {
                printf("[");
                for (int w = 0; w < shape->w; w++) {
                    printf("%f ", at(n,c,h,w)); 
                }
                printf("]\n");
            }
            printf("]\n");
        }
        printf("]\n");
    }
}

std::shared_ptr<Tensor> Tensor::zeros(Shape* shape_) {
    int size_ = accum_size(shape_);
    float* data_ = (float*) mkl_calloc(size_, sizeof(float), MALLOC_ALIGN);

    return std::shared_ptr<Tensor>(new Tensor(data_, shape_, size_));
}

std::shared_ptr<Tensor> Tensor::random(Shape* shape_, float low, float high) {
    int size_ = accum_size(shape_);
    float* data_ = (float*) mkl_malloc(size_ * sizeof(float), MALLOC_ALIGN);
    global_rand->populate(size_, data_, low, high);

    return std::shared_ptr<Tensor>(new Tensor(data_, shape_, size_));
}

void Tensor::bias_cpy(float* bias, float* dest, int bias_size, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        std::memcpy(bias, dest+i*bias_size*sizeof(float), bias_size * sizeof(float));
    }
}
