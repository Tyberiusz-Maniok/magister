#pragma once
#include <valarray>
#include "rng.h"

namespace lamp {

class Tensor {
    public:
        float* data;
        int* shape;
        int size;
        int* strides;
        int rank;
        bool requires_grad;
        // Tensor& grad;

        Tensor(float* data, int* shape, int rank, bool requires_grad=true);
        Tensor(float* data, int* shape, int rank, int size, bool requires_grad=true);
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

        float operator[](int idx);
        Tensor operator[](std::slice idx);
        Tensor operator[](std::gslice idx);

        float dot(Tensor& other);
        Tensor& matmul(Tensor& other);
        Tensor tranpose(int ax1, int ax2);
        Tensor stack();
        Tensor unstack(int* axes);
        Tensor reshape(int* shape);
        Tensor backwards();
        Tensor flip();

        int* shift_slice(Tensor& tensor, int axis, int shift);
    
        static Tensor* zeros(int* shape_, int rank_);
        static Tensor* random(int* shape_, int rank_, RandomGen& rng, float low = -1, float high = 1);
};

}
