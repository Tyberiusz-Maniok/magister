#include "tensor.h"
#include <omp.h>
#include <stdlib.h>
#include <mkl.h>
#include <mkl_vsl.h>
#include <cstring>

#define MALLOC_ALIGN 32

using namespace lamp;

int accum_size(int* shape, int rank) {
    int size = 1;
    for (int i = 0; i < rank; i++) {
        size *= *(shape+i);
    }
    return size;
}

void set_strides(int* shape, int* strides, int rank) {
    int stride = 1;
    for (int i = rank - 1; i >= 0; i++) {
        *(strides + i) = stride;
        stride *= *(shape + i);
    }
}

Tensor::Tensor(float* data, int* shape, int rank, bool requires_grad) : data(data), shape(shape),
    rank(rank), requires_grad(requires_grad) {
    this->size = accum_size(shape, rank);

    this->strides = (int*) mkl_malloc(rank * sizeof(int), MALLOC_ALIGN);
    set_strides(shape, this->strides, rank);
}

Tensor::Tensor(float* data, int* shape, int rank, int size, bool requires_grad) : data(data), shape(shape),
    rank(rank), size(size), requires_grad(requires_grad) {

    this->strides = (int*) mkl_malloc(rank * sizeof(int), MALLOC_ALIGN);
    set_strides(shape, this->strides, rank);
}

Tensor::Tensor(Tensor& other) : size(other.size), rank(other.rank), requires_grad(other.requires_grad){
    this->data = (float*) mkl_malloc(other.size * sizeof(float), MALLOC_ALIGN);
    std::memcpy(other.data, this->data, other.size * sizeof(float));

    this->shape = (int*) mkl_malloc(other.rank * sizeof(int), MALLOC_ALIGN);
    std::memcpy(other.shape, this->shape, other.rank * sizeof(int));

    this->strides = (int*) mkl_malloc(other.rank * sizeof(int), MALLOC_ALIGN);
    std::memcpy(other.strides, this->strides, other.rank * sizeof(int));
}

Tensor::~Tensor() {
    mkl_free(this->data);
    mkl_free(this->shape);
    mkl_free(this->strides);
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
    #pragma omp parallel for
    for (int i=0; i < this->size; i++) {
        *(this->data+i) += *(other.data+i);
    }
    return *this;
}

Tensor& Tensor::operator-=(Tensor& other) {
    #pragma omp parallel for
    for (int i=0; i < this->size; i++) {
        *(this->data+i) -= *(other.data+i);
    }
    return *this;
}

Tensor& Tensor::operator*=(Tensor& other) {
    #pragma omp parallel for
    for (int i=0; i < this->size; i++) {
        *(this->data+i) *= *(other.data+i);
    }
    return *this;
}

Tensor& Tensor::operator/=(Tensor& other) {
    #pragma omp parallel for
    for (int i=0; i < this->size; i++) {
        *(this->data+i) /= *(other.data+i);
    }
    return *this;
}

float Tensor::operator[](int idx) {
    return *(this->data+idx);
}

float Tensor::dot(Tensor& other) {
    return cblas_sdot(this->size, this->data, 1, other.data, 1);
}

Tensor& Tensor::matmul(Tensor& other) {
    int m = *(this->shape);
    int k = *(this->shape+1);
    int n = *(other.shape+1);
    float* result = (float*) mkl_malloc(m * n * sizeof(float), MALLOC_ALIGN);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, this->data, k, other.data, n, 0, result, n);

    int* shape = (int*) mkl_malloc(2 * sizeof(int), MALLOC_ALIGN);
    *shape = n;
    *(shape+1) = m;
    return *(new Tensor(result, shape, 2, m*n));
}

Tensor* Tensor::zeros(int* shape, int rank) {
    int size = accum_size(shape, rank);
    float* data = (float*) mkl_calloc(size, sizeof(float), MALLOC_ALIGN);

    return new Tensor(data, shape, rank, size);
}

Tensor* Tensor::random(int* shape, int rank, RandomGen& rng, float low, float high) {
    int size = accum_size(shape, rank);
    float* data = (float*) mkl_malloc(size * sizeof(float), MALLOC_ALIGN);
    rng.populate(size, data, low, high);

    return new Tensor(data, shape, rank, size);
}
