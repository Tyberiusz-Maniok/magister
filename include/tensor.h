#pragma once
#include "rng.h"
#include "mkl.h"
#include <memory>

namespace lamp {

struct Shape {
    int n, c, h, w;

    Shape(int n, int c, int h, int w) : n(n), c(c), h(h), w(w) {}
    Shape(Shape& other) : n(other.n), c(other.c), h(other.h), w(other.w) {}
};



class Tensor {
    private:
        void set_strides(Shape* shape);
    public:
        float* data;
        Shape* shape;
        int size;
        Shape* strides;
        int rank;

        Tensor(float* data, Shape* shape);
        Tensor(float* data, Shape* shape, int size);
        Tensor(Tensor& other);
        Tensor(std::shared_ptr<Tensor> other);
        ~Tensor();

        Tensor& operator=(Tensor& other);

        Tensor& operator+(Tensor& other);
        Tensor& operator-(Tensor& other);
        Tensor& operator*(Tensor& other);
        Tensor& operator/(Tensor& other);

        Tensor& operator+=(Tensor& other);
        Tensor& operator-=(Tensor& other);
        Tensor& operator*=(Tensor& other);
        Tensor& operator/=(Tensor& other);

        Tensor& operator*=(float other);

        void mulsub(std::shared_ptr<Tensor> other, float mul);

        int flat_index(int n, int c, int h, int w);
        float at(int n, int c, int h, int w);
        float operator[](int idx);

        float dot(Tensor& other);
        std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> other, std::shared_ptr<Tensor> bias = nullptr, CBLAS_TRANSPOSE transa = CblasNoTrans, CBLAS_TRANSPOSE transb = CblasNoTrans);
        std::shared_ptr<Tensor> batched_matmul(std::shared_ptr<Tensor> other, std::shared_ptr<Tensor> bias = nullptr, CBLAS_TRANSPOSE transa = CblasNoTrans, CBLAS_TRANSPOSE transb = CblasNoTrans);
        std::shared_ptr<Tensor> avg_grad();
        void reshape(Shape* shape);
        void reshape(int n, int c, int h, int w);
        float sum();
        float sum2d(int n, int c);
        float avg();
        float avg2d(int n, int c);
        float variance();
        float variance2d(int n, int c);
        float variance_from_avg(float avg);
        float variance_from_avg2d(int n, int c, float avg);

        void print();
        void print_shape();
    
        static std::shared_ptr<Tensor> zeros(Shape* shape_);
        static std::shared_ptr<Tensor> random(Shape* shape_, float low = -1, float high = 1);
        static void bias_cpy(float* bias, float* dest, int bias_size, int n);
};

typedef std::shared_ptr<Tensor> TensorP;

// #pragma omp declare mapper(Tensor t) map(to: t, t.data[0:t.size])

}
