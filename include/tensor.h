#pragma once
#include <valarray>
#include "rng.h"

namespace lamp {

struct Shape {
    int n, c, h, w;

    Shape(int n, int c, int h, int w) : n(n), c(c), h(h), w(w) {}
    Shape(Shape& other) : n(other.n), c(other.c), h(other.h), w(other.w) {}
};

class Tensor {
    public:
        float* data;
        Shape* shape;
        int size;
        Shape* strides;
        int rank;
        bool requires_grad;
        // Tensor& grad;

        Tensor(float* data, Shape* shape, bool requires_grad=true);
        Tensor(float* data, Shape* shape, int size, bool requires_grad=true);
        Tensor(Tensor& other);
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

        int flat_index(int n, int c, int h, int w);
        float at(int n, int c, int h, int w);
        float operator[](int idx);
        // Tensor operator[](std::slice idx);
        // Tensor operator[](std::gslice idx);

        float dot(Tensor& other);
        Tensor& matmul(Tensor& other, Tensor* bias = nullptr);
        void reshape(Shape* shape);
        void reshape(int n, int c, int h, int w);
        Tensor backwards();

        void print();
    
        static Tensor* zeros(Shape* shape_);
        static Tensor* random(Shape* shape_, RandomGen& rng, float low = -1, float high = 1);
};

}
