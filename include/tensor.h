#pragma once
#include <valarray>

class Tensor {
    public:
        float* data;
        int* shape;
        int size;
        int* strides;
        int rank;
        bool requires_grad;

        Tensor(float* data, int* shape, int rank, bool requires_grad=true);
        Tensor(float* data, int* shape, int rank, int size, bool requires_grad=true);
        Tensor(const Tensor& other);
        ~Tensor();

        Tensor& operator=(const Tensor& other);

        Tensor& operator+(const Tensor& other);
        Tensor& operator-(const Tensor& other);
        Tensor& operator*(const Tensor& other);
        Tensor& operator/(const Tensor& other);

        Tensor& operator+=(const Tensor& other);
        Tensor& operator-=(const Tensor& other);
        Tensor& operator*=(const Tensor& other);
        Tensor& operator/=(const Tensor& other);

        float operator[](int idx);
        Tensor operator[](std::slice idx);
        Tensor operator[](std::gslice idx);

        float dot(const Tensor& other);
        Tensor& matmul(const Tensor& other);
        Tensor tranpose(int ax1, int ax2);
        Tensor stack();
        Tensor unstack(int* axes);
        Tensor reshape(int* shape);
        Tensor backwards();
        Tensor flip();

        int* shift_slice(const Tensor& tensor, int axis, int shift);
    
        static Tensor& zeros(int* shape, int rank);
        static Tensor& random(int* shape, int rank, float low = -1, float high = 1);
};
