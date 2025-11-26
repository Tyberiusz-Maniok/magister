// CRITICAL ORDER: Include all C++ standard library headers BEFORE CUDA headers
// to avoid __noinline__ conflicts between libstdc++ and CUDA's host_defines.h
#include <memory>
#include <cstring>
#include <cstdio>
#include <stdlib.h>
#include <random>
#include <string>

// Workaround for CUDA 12.6 + NVHPC 24.9 compatibility
#include "cuda_nvhpc_compat.h"

// Now safe to include CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Then our headers
#include "tensor.h"
#include <omp.h>
#include "consts.h"

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
    this->data = (float*) aligned_alloc(MALLOC_ALIGN, other.size * sizeof(float));
    std::memcpy(other.data, this->data, other.size * sizeof(float));
}

Tensor::Tensor(std::shared_ptr<Tensor> other) : size(other->size), shape(new Shape(*(other->shape))), strides(new Shape(*(other->strides))) {
    this->data = (float*) aligned_alloc(MALLOC_ALIGN, other->size * sizeof(float));
    std::memcpy(other->data, this->data, other->size * sizeof(float));
}

Tensor::~Tensor() {
    free(this->data);
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

std::shared_ptr<Tensor> Tensor::add(std::shared_ptr<Tensor> other) {
    Tensor* result = new Tensor(*this);
    #pragma omp parallel for simd
    for (int i=0; i < this->size; i++) {
        *(data+i) += *(other->data+i);
    }
    return std::shared_ptr<Tensor>(result);
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
    float result = 0.0f;
    for (int i = 0; i < size; ++i) {
        result += data[i] * other.data[i];
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> other, std::shared_ptr<Tensor> bias, cublasOperation_t transa, cublasOperation_t transb) {
    // Original row-major dimensions
    int m = this->shape->h;
    int k = this->shape->w;
    int n = other->shape->w;

    int lda = k;
    int ldb = n;
    int ldc = n;

    if (transa == CUBLAS_OP_T) {
        m = k;
        k = this->shape->h;
        lda = m;
    }
    if (transb == CUBLAS_OP_T) {
        n = other->shape->h;
        ldc = n;
        ldb = k;
    }

    float* result = (float*) aligned_alloc(MALLOC_ALIGN, m * n * sizeof(float));
    float beta_val = 0;
    float alpha = 1;
    if (bias != nullptr) {
        Tensor::bias_cpy(bias->data, result, bias->size, this->shape->n);
        beta_val = 1;
    }
    
    // Convert row-major to column-major by swapping operands and dimensions
    // Row-major: C = op(A) * op(B) becomes Column-major: C^T = op(B)^T * op(A)^T
    // Swap A<->B, transa<->transb, m<->n, lda<->ldb
    cublasOperation_t transa_col = transb;
    cublasOperation_t transb_col = transa;
    
    // Allocate device memory using CUDA
    float* d_tdata;
    float* d_odata;
    float* d_result;
    cudaMalloc(&d_tdata, this->size * sizeof(float));
    cudaMalloc(&d_odata, other->size * sizeof(float));
    cudaMalloc(&d_result, m * n * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_tdata, this->data, this->size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_odata, other->data, other->size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result, m * n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Use CUBLAS with device pointers
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS create failed: %d\n", status);
        return nullptr;
    }
    
    // CUBLAS column-major call: swap A<->B, swap m<->n, swap lda<->ldb
    status = cublasSgemm(handle, transa_col, transb_col, n, m, k, &alpha, d_odata, ldb, d_tdata, lda, &beta_val, d_result, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS sgemm failed: %d\n", status);
    }
    
    // CRITICAL: Synchronize before destroying handle (CUBLAS ops are async)
    cudaError_t cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        printf("CUDA sync failed: %s\n", cudaGetErrorString(cuda_status));
    }
    
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS destroy failed: %d\n", status);
    }
    
    // Copy result back to host
    cudaMemcpy(result, d_result, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_tdata);
    cudaFree(d_odata);
    cudaFree(d_result);

    return std::shared_ptr<Tensor>(new Tensor(result, new Shape(1, 1, m, n), m*n));
}

std::shared_ptr<Tensor> Tensor::batched_matmul(std::shared_ptr<Tensor> other, std::shared_ptr<Tensor> bias, cublasOperation_t transa, cublasOperation_t transb) {
    // Original row-major dimensions
    int m = this->shape->h;
    int k = this->shape->w;
    int n = other->shape->w;

    int lda = k;
    int ldb = n;
    int ldc = n;

    if (transa == CUBLAS_OP_T) {
        m = k;
        k = this->shape->h;
        lda = m;
    }
    if (transb == CUBLAS_OP_T) {
        n = other->shape->h;
        ldc = n;
        ldb = k;
    }

    float* result = (float*) aligned_alloc(MALLOC_ALIGN, m * n * other->shape->n * sizeof(float));
    float beta_val = 0;
    float alpha = 1;
    if (bias != nullptr) {
        Tensor::bias_cpy(bias->data, result, bias->size, this->shape->n);
        beta_val = 1;
    }

    int in_stride = other->strides->n;
    int out_stride = m * n;
    
    // Convert row-major to column-major by swapping operands and dimensions
    // Row-major: C = op(A) * op(B) becomes Column-major: C^T = op(B)^T * op(A)^T
    // Swap A<->B, transa<->transb, m<->n, lda<->ldb
    cublasOperation_t transa_col = transb;
    cublasOperation_t transb_col = transa;

    // Allocate device memory using CUDA
    float* d_tdata;
    float* d_odata;
    float* d_result;
    cudaMalloc(&d_tdata, this->size * sizeof(float));
    cudaMalloc(&d_odata, other->size * sizeof(float));
    cudaMalloc(&d_result, m * n * other->shape->n * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_tdata, this->data, this->size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_odata, other->data, other->size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result, m * n * other->shape->n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Use CUBLAS with device pointers
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS create failed: %d\n", status);
        return nullptr;
    }
    
    // Note: Using sequential loop - CUBLAS calls are async on GPU internally
    // For true batched operations, consider using cublasSgemmStridedBatched
    for (int i = 0; i < other->shape->n; i++) {
        // CUBLAS column-major call: swap A<->B, swap m<->n, swap lda<->ldb
        status = cublasSgemm(handle, transa_col, transb_col, n, m, k, &alpha, d_odata+(in_stride*i), ldb, d_tdata, lda, &beta_val, d_result+(out_stride*i), ldc);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("CUBLAS sgemm failed at iteration %d: %d\n", i, status);
        }
    }
    
    // CRITICAL: Synchronize before destroying handle (CUBLAS ops are async)
    cudaError_t cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        printf("CUDA sync failed: %s\n", cudaGetErrorString(cuda_status));
    }
    
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS destroy failed: %d\n", status);
    }
    
    // Copy result back to host
    cudaMemcpy(result, d_result, m * n * other->shape->n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_tdata);
    cudaFree(d_odata);
    cudaFree(d_result);

    return std::shared_ptr<Tensor>(new Tensor(result, new Shape(other->shape->n, 1, m, n), m * n * other->shape->n));
}

std::shared_ptr<Tensor> Tensor::avg_grad() {
    if (shape->n == 1) {
        return std::shared_ptr<Tensor>(new Tensor(*this));
    }
    float* result = (float*) aligned_alloc(MALLOC_ALIGN, shape->c * shape->h * shape->w * sizeof(float));
    memset(result, 0, shape->c * shape->h * shape->w * sizeof(float));

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
    float* data_ = (float*) aligned_alloc(MALLOC_ALIGN, size_ * sizeof(float));
    memset(data_, 0, size_ * sizeof(float));

    return std::shared_ptr<Tensor>(new Tensor(data_, shape_, size_));
}

std::shared_ptr<Tensor> Tensor::random(Shape* shape_, float low, float high) {
    int size_ = accum_size(shape_);
    float* data_ = (float*) aligned_alloc(MALLOC_ALIGN, size_ * sizeof(float));
    global_rand->populate(size_, data_, low, high);

    return std::shared_ptr<Tensor>(new Tensor(data_, shape_, size_));
}

void Tensor::bias_cpy(float* bias, float* dest, int bias_size, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        std::memcpy(bias, dest+i*bias_size*sizeof(float), bias_size * sizeof(float));
    }
}
