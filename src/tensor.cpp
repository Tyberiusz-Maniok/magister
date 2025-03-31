#include "tensor.h"
#include <omp.h>
#include <stdlib.h>
#include <mkl.h>
#include <mkl_vsl.h>
#include <cstring>

#include <cstdio>

#define MALLOC_ALIGN 32

using namespace lamp;

int accum_size(Shape* shape) {
    return shape->n * shape->c * shape->h * shape->w;
}

Shape* set_strides(Shape* shape) {
    return new Shape(
        shape->c * shape->h * shape->w,
        shape->h * shape->w,
        shape->w,
        1
    );
}

Tensor::Tensor(float* data, Shape* shape, bool requires_grad) : data(data), shape(shape),
    requires_grad(requires_grad) {
    this->size = accum_size(shape);
    this->strides = set_strides(shape);
}

Tensor::Tensor(float* data, Shape* shape, int size, bool requires_grad) : data(data), shape(shape),
    size(size), requires_grad(requires_grad) {
    set_strides(shape);
}

Tensor::Tensor(Tensor& other) : size(other.size), requires_grad(other.requires_grad),
    shape(new Shape(*(other.shape))), strides(new Shape(*(other.strides))) {
    this->data = (float*) mkl_malloc(other.size * sizeof(float), MALLOC_ALIGN);
    std::memcpy(other.data, this->data, other.size * sizeof(float));
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

Tensor& Tensor::matmul(Tensor& other, Tensor* bias) {
    int m = this->shape->h;
    int k = this->shape->w;
    int n = other.shape->w;

    float* result = (float*) mkl_malloc(m * n * sizeof(float), MALLOC_ALIGN);
    float beta = 0;
    if (bias != nullptr) {
        std::memcpy(bias, result, m * n * sizeof(float));
        beta = 1;
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, this->data, k, other.data, n, beta, result, n);

    Shape* shape = new Shape(1, 1, n ,m);
    return *(new Tensor(result, shape, m*n));
}

Tensor* Tensor::zeros(Shape* shape_) {
    int size_ = accum_size(shape_);
    float* data_ = (float*) mkl_calloc(size_, sizeof(float), MALLOC_ALIGN);

    return new Tensor(data_, shape_, size_);
}

Tensor* Tensor::random(Shape* shape_, RandomGen& rng, float low, float high) {
    int size_ = accum_size(shape_);
    float* data_ = (float*) mkl_malloc(size_ * sizeof(float), MALLOC_ALIGN);
    rng.populate(size_, data_, low, high);

    return new Tensor(data_, shape_, size_);
}
