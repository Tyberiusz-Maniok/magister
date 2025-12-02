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

// Standalone helper functions for GPU kernels (avoid pointer dereferencing)
// NCHW format: strides are (C*H*W, H*W, W, 1)
inline int compute_flat_index(int n, int c, int h, int w, int stride_n, int stride_c, int stride_h, int stride_w) {
    return n * stride_n + c * stride_c + h * stride_h + w * stride_w;
}

inline float tensor_at(const float* data, int n, int c, int h, int w, int stride_n, int stride_c, int stride_h, int stride_w) {
    return data[n * stride_n + c * stride_c + h * stride_h + w * stride_w];
}

void Tensor::set_strides(Shape* shape) {
    this->strides = new Shape(
        shape->c * shape->h * shape->w,
        shape->h * shape->w,
        shape->w,
        1
    );
}

Tensor::Tensor(float* data, Shape* shape) : data(data), shape(shape), d_data(nullptr) {
    this->size = accum_size(shape);
    set_strides(shape);
    
    // Allocate device memory with OpenMP
    int device_num = omp_get_default_device();
    this->d_data = (float*) omp_target_alloc(this->size * sizeof(float), device_num);
    if (this->d_data == nullptr) {
        printf("ERROR: omp_target_alloc failed for size %d (%ld MB)\n", this->size, (this->size * sizeof(float)) / (1024*1024));
        return;
    }
    
    // Copy data to device
    int host_num = omp_get_initial_device();
    int result = omp_target_memcpy(this->d_data, this->data, this->size * sizeof(float), 
                      0, 0, device_num, host_num);
    if (result != 0) {
        printf("ERROR: omp_target_memcpy failed with code %d\n", result);
    }
}

Tensor::Tensor(float* data, Shape* shape, int size) : data(data), shape(shape), size(size), d_data(nullptr) {
    set_strides(shape);
    
    // Allocate device memory with OpenMP
    int device_num = omp_get_default_device();
    this->d_data = (float*) omp_target_alloc(size * sizeof(float), device_num);
    if (this->d_data == nullptr) {
        printf("ERROR: omp_target_alloc failed for size %d\n", size);
        return;
    }
    
    // Copy data to device
    int host_num = omp_get_initial_device();
    int result = omp_target_memcpy(this->d_data, this->data, size * sizeof(float), 
                      0, 0, device_num, host_num);
    if (result != 0) {
        printf("ERROR: omp_target_memcpy failed with code %d\n", result);
    }
}

Tensor::Tensor(Tensor& other) : size(other.size), shape(new Shape(*(other.shape))), strides(new Shape(*(other.strides))), d_data(nullptr) {
    this->data = (float*) aligned_alloc(MALLOC_ALIGN, other.size * sizeof(float));
    std::memcpy(other.data, this->data, other.size * sizeof(float));
    
    // Allocate device memory and copy from other's device memory
    int device_num = omp_get_default_device();
    this->d_data = (float*) omp_target_alloc(other.size * sizeof(float), device_num);
    if (this->d_data != nullptr && other.d_data != nullptr) {
        omp_target_memcpy(this->d_data, other.d_data, other.size * sizeof(float),
                          0, 0, device_num, device_num);
    }
}

Tensor::Tensor(std::shared_ptr<Tensor> other) : size(other->size), shape(new Shape(*(other->shape))), strides(new Shape(*(other->strides))), d_data(nullptr) {
    this->data = (float*) aligned_alloc(MALLOC_ALIGN, other->size * sizeof(float));
    std::memcpy(other->data, this->data, other->size * sizeof(float));
    
    // Allocate device memory and copy from other's device memory
    int device_num = omp_get_default_device();
    this->d_data = (float*) omp_target_alloc(other->size * sizeof(float), device_num);
    if (this->d_data != nullptr && other->d_data != nullptr) {
        omp_target_memcpy(this->d_data, other->d_data, other->size * sizeof(float),
                          0, 0, device_num, device_num);
    }
}

Tensor::~Tensor() {
    // Free device memory
    if (this->d_data != nullptr) {
        int device_num = omp_get_default_device();
        omp_target_free(this->d_data, device_num);
    }
    
    // Free host memory
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
    float* this_dev = d_data;
    float* other_dev = other.d_data;
    int data_size = size;
    #pragma omp target teams distribute parallel for simd is_device_ptr(this_dev, other_dev)
    for (int i=0; i < data_size; i++) {
        *(this_dev+i) += *(other_dev+i);
    }
    return *this;
}

Tensor& Tensor::operator-=(Tensor& other) {
    float* this_dev = d_data;
    float* other_dev = other.d_data;
    int data_size = size;
    #pragma omp target teams distribute parallel for simd is_device_ptr(this_dev, other_dev)
    for (int i=0; i < data_size; i++) {
        *(this_dev+i) -= *(other_dev+i);
    }
    return *this;
}

Tensor& Tensor::operator*=(Tensor& other) {
    float* this_dev = d_data;
    float* other_dev = other.d_data;
    int data_size = size;
    #pragma omp target teams distribute parallel for simd is_device_ptr(this_dev, other_dev)
    for (int i=0; i < data_size; i++) {
        *(this_dev+i) *= *(other_dev+i);
    }
    return *this;
}

Tensor& Tensor::operator/=(Tensor& other) {
    float* this_dev = d_data;
    float* other_dev = other.d_data;
    int data_size = size;
    #pragma omp target teams distribute parallel for simd is_device_ptr(this_dev, other_dev)
    for (int i=0; i < data_size; i++) {
        *(this_dev+i) /= *(other_dev+i);
    }
    return *this;
}

Tensor& Tensor::operator*=(float other) {
    float* this_dev = d_data;
    int data_size = size;
    #pragma omp target teams distribute parallel for simd is_device_ptr(this_dev)
    for (int i=0; i < data_size; i++) {
        *(this_dev+i) *= other;
    }
    return *this;
}

std::shared_ptr<Tensor> Tensor::add(std::shared_ptr<Tensor> other) {
    Tensor* result = new Tensor(*this);
    float* this_dev = d_data;
    float* other_dev = other->d_data;
    int data_size = size;
    #pragma omp target teams distribute parallel for simd is_device_ptr(this_dev, other_dev)
    for (int i=0; i < data_size; i++) {
        *(this_dev+i) += *(other_dev+i);
    }
    return std::shared_ptr<Tensor>(result);
}

void Tensor::mulsub(std::shared_ptr<Tensor> other, float mul) {
    float* this_dev = d_data;
    float* other_dev = other->d_data;
    int data_size = size;
    #pragma omp target teams distribute parallel for simd is_device_ptr(this_dev, other_dev)
    for (int i = 0; i < data_size; i++) {
        *(this_dev+i) -= *(other_dev+i) * mul;
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
    
    // Use existing device pointers from omp_target_alloc
    float* d_tdata = this->d_data;
    float* d_odata = other->d_data;
    
    // Allocate device memory for result using OpenMP
    int device_num = omp_get_default_device();
    float* d_result = (float*) omp_target_alloc(m * n * sizeof(float), device_num);
    
    // Copy bias data to device if needed
    if (beta_val != 0) {
        int host_num = omp_get_initial_device();
        omp_target_memcpy(d_result, result, m * n * sizeof(float), 0, 0, device_num, host_num);
    }
    
    // Create CUBLAS handle - shares CUDA context with OpenMP due to -cuda flag
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
    
    // CRITICAL: Synchronize to ensure CUBLAS operations complete before proceeding
    cudaDeviceSynchronize();
    
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS destroy failed: %d\n", status);
    }
    
    // Copy result back to host
    int host_num = omp_get_initial_device();
    omp_target_memcpy(result, d_result, m * n * sizeof(float), 0, 0, host_num, device_num);
    
    // Free device result memory
    omp_target_free(d_result, device_num);

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

    // Use existing device pointers from omp_target_alloc
    float* d_tdata = this->d_data;
    float* d_odata = other->d_data;
    
    // Allocate device memory for result using OpenMP
    int device_num = omp_get_default_device();
    float* d_result = (float*) omp_target_alloc(m * n * other->shape->n * sizeof(float), device_num);
    
    // Copy bias data to device if needed
    if (beta_val != 0) {
        int host_num = omp_get_initial_device();
        omp_target_memcpy(d_result, result, m * n * other->shape->n * sizeof(float), 0, 0, device_num, host_num);
    }
    
    // Create CUBLAS handle - shares CUDA context with OpenMP due to -cuda flag
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
    
    // CRITICAL: Synchronize to ensure all CUBLAS operations complete before proceeding
    cudaDeviceSynchronize();
    
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS destroy failed: %d\n", status);
    }
    
    // Copy result back to host
    int host_num = omp_get_initial_device();
    omp_target_memcpy(result, d_result, m * n * other->shape->n * sizeof(float), 0, 0, host_num, device_num);
    
    // Free device result memory
    omp_target_free(d_result, device_num);

    return std::shared_ptr<Tensor>(new Tensor(result, new Shape(other->shape->n, 1, m, n), m * n * other->shape->n));
}

std::shared_ptr<Tensor> Tensor::avg_grad() {
    if (shape->n == 1) {
        return std::shared_ptr<Tensor>(new Tensor(*this));
    }
    float* result = (float*) aligned_alloc(MALLOC_ALIGN, shape->c * shape->h * shape->w * sizeof(float));
    memset(result, 0, shape->c * shape->h * shape->w * sizeof(float));

    int batch_size = shape->n;
    int elem_per_batch = size / batch_size;
    int stride_n = strides->n;
    float* this_dev = d_data;
    
    // Allocate result on device
    int device_num = omp_get_default_device();
    float* result_dev = (float*) omp_target_alloc(elem_per_batch * sizeof(float), device_num);
    int host_num = omp_get_initial_device();
    omp_target_memcpy(result_dev, result, elem_per_batch * sizeof(float), 0, 0, device_num, host_num);
    
    #pragma omp target teams distribute parallel for is_device_ptr(this_dev, result_dev)
    for (int i = 0; i < elem_per_batch; i++) {
        for (int n = 0; n < batch_size; n++) {
            *(result_dev+i) += *(this_dev+i+(n*stride_n));
        }
        *(result_dev+i) /= batch_size;
    }
    
    // Copy result back to host
    omp_target_memcpy(result, result_dev, elem_per_batch * sizeof(float), 0, 0, host_num, device_num);
    omp_target_free(result_dev, device_num);

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
    float* this_dev = d_data;
    int data_size = size;
    #pragma omp target teams distribute parallel for simd reduction(+:result) is_device_ptr(this_dev)
    for (int i = 0; i < data_size; i++) {
        result += *(this_dev+i);
    }
    return result;
}

float Tensor::avg() {
    return sum() / size;
}

float Tensor::variance() {
    float result = 0;
    float mean = avg();
    float* this_dev = d_data;
    int data_size = size;
    #pragma omp target teams distribute parallel for simd reduction(+:result) is_device_ptr(this_dev)
    for (int i = 0; i < data_size; i++) {
        float diff = *(this_dev+i) - mean;
        result += diff * diff;
    }
    return result / size;
}

float Tensor::variance_from_avg(float avg) {
    float result = 0;
    float* this_dev = d_data;
    int data_size = size;
    #pragma omp target teams distribute parallel for simd reduction(+:result) is_device_ptr(this_dev)
    for (int i = 0; i < data_size; i++) {
        result += *(this_dev+i) - avg;
    }
    return result / size;
}

float Tensor::sum2d(int n, int c) {
    float result = 0;
    int start_idx = n * strides->n + c * strides->c;
    int num_elem = shape->h * shape->w;
    float* this_dev = d_data;
    #pragma omp target teams distribute parallel for simd reduction(+:result) is_device_ptr(this_dev)
    for (int i = start_idx; i < start_idx + num_elem; i++) {
        result += *(this_dev+i);
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
    int num_elem = shape->h * shape->w;
    float* this_dev = d_data;
    #pragma omp target teams distribute parallel for simd reduction(+:result) is_device_ptr(this_dev)
    for (int i = start_idx; i < start_idx + num_elem; i++) {
        float diff = *(this_dev+i) - mean;
        result += diff * diff;
    }
    return result / size;
}

float Tensor::variance_from_avg2d(int n, int c, float avg) {
    float result = 0;
    int start_idx = n * strides->n + c * strides->c;
    int num_elem = shape->h * shape->w;
    float* this_dev = d_data;
    #pragma omp target teams distribute parallel for simd reduction(+:result) is_device_ptr(this_dev)
    for (int i = start_idx; i < start_idx + num_elem; i++) {
        float diff = *(this_dev+i) - avg;
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
    // Allocate temporary device memory for host pointers
    int device_num = omp_get_default_device();
    int host_num = omp_get_initial_device();
    int total_size = bias_size * n;
    
    float* bias_dev = (float*) omp_target_alloc(bias_size * sizeof(float), device_num);
    float* dest_dev = (float*) omp_target_alloc(total_size * sizeof(float), device_num);
    
    omp_target_memcpy(bias_dev, bias, bias_size * sizeof(float), 0, 0, device_num, host_num);
    
    #pragma omp target teams distribute parallel for simd is_device_ptr(bias_dev, dest_dev)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < bias_size; j++) {
            *(dest_dev + i * bias_size + j) = *(bias_dev + j);
        }
    }
    
    omp_target_memcpy(dest, dest_dev, total_size * sizeof(float), 0, 0, host_num, device_num);
    omp_target_free(bias_dev, device_num);
    omp_target_free(dest_dev, device_num);
}
