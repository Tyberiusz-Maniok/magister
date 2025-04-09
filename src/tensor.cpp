#include "tensor.h"
#include <omp.h>
#include <stdlib.h>
#include <mkl.h>
#include <mkl_vsl.h>
#include "consts.h"
#include <cstring>

#include <cstdio>

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
    this->strides = set_strides(shape);
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

int Tensor::flat_index(int n, int c, int h, int w) {
    // printf("idx = %i", n * this->strides->n + c * this->strides->c + h * this->strides->h + w * this->strides->w);
    return n * this->strides->n + c * this->strides->c + h * this->strides->h + w * this->strides->w;
}

float Tensor::at(int n, int c, int h, int w) {
    return *(this->data+flat_index(n, c, h, w));
}

float Tensor::operator[](int idx) {
    return *(this->data+idx);
}

float Tensor::dot(Tensor& other) {
    return cblas_sdot(this->size, this->data, 1, other.data, 1);
}

Tensor& Tensor::matmul(Tensor& other, Tensor* bias, CBLAS_TRANSPOSE transa) {
    int m = this->shape->h;
    int k = this->shape->w;
    int n = other.shape->w;

    float* result = (float*) mkl_malloc(m * n * sizeof(float), MALLOC_ALIGN);
    float beta = 0;
    if (bias != nullptr) {
        std::memcpy(bias->data, result, m * n * sizeof(float));
        beta = 1;
    }

    cblas_sgemm(CblasRowMajor, transa, CblasNoTrans, m, n, k, 1, this->data, k, other.data, n, beta, result, n);

    return *(new Tensor(result, new Shape(1, 1, n ,m), m*n));
}

void Tensor::reshape(int n, int c, int h, int w) {
    this->shape->n = n;
    this->shape->c = c;
    this->shape->h = h;
    this->shape->w = w;
    // delete this->strides;
    this->strides = set_strides(this->shape);
}

float Tensor::sum() {
    float result = 0;
    #pragma omp parallel for simd reduction(+:result)
    for (int i = 0; i < this->size; i++) {
        result += *(this->data+i);
    }
    return result;
}

void Tensor::print() {
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
