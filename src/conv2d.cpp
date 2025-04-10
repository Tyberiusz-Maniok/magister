#include "conv2d.h"
#include <mkl.h>
#include <omp.h>
#include "consts.h"

using namespace lamp;

Conv2d::Conv2d(int input, int output, int kernel, int stride, RandomGen& rng, activ_fn activation_fn) : in_c(input), out_c(output),
    activation_fn(activation_fn), k(kernel), stride(stride) {
    this->filters = Tensor::random(new Shape(output, input, kernel, kernel), rng);
    this->bias = nullptr;
}

Conv2d::~Conv2d() {
    delete this->filters;
    delete this->bias;
    delete this->input_col;
}

Tensor& Conv2d::im2col(Tensor& x) {
    float* col_data = (float*) mkl_calloc(x.shape->n * out_h * out_w * x.shape->c * this->k * this->k, sizeof(float), MALLOC_ALIGN);
    Shape* col_shape = new Shape(x.shape->n, x.shape->c, k*k, out_h*out_w);
    Tensor* col = new Tensor(col_data, col_shape);

    #pragma omp parallel for simd collapse(6)
    for (int n = 0; n < x.shape->n; n++) {
        for (int c = 0; c < x.shape->c; c++) {
            for (int h = 0; h < this->out_h; h++) {
                for (int w = 0; w < this->out_w; w++) {
                    for (int kh = 0; kh < this->k; kh++) {
                        for (int kw = 0; kw < this->k; kw++) {
                            *(col_data+col->flat_index(n, c, kh * k + kw, h * out_w + w)) = x.at(n, c, h * stride + kh, w * stride + kw);
                        }
                    }
                }
            }
        }
    }

    return *col;
}

Tensor& Conv2d::col2im(Tensor& x, Shape& shape) {
    float* im_data = (float*) mkl_calloc(shape.n * shape.c * shape.h * shape.w, sizeof(float), MALLOC_ALIGN);
    Tensor* im = new Tensor(im_data, new Shape(shape));
    
    #pragma omp parallel for simd collapse(6)
    for (int n = 0; n < x.shape->n; n++) {
        for (int c = 0; c < x.shape->c; c++) {
            for (int h = 0; h < this->out_h; h++) {
                for (int w = 0; w < this->out_w; w++) {
                    for (int kh = 0; kh < this->k; kh++) {
                        for (int kw = 0; kw < this->k; kw++) {
                            *(im_data+im->flat_index(n, c, h * stride + kh, w * stride + kw)) += x.at(n, c, h * this->out_h + kh, w * this->out_w + kw);
                        }
                    }
                }
            }
        }
    }
    return *im;
}

Tensor& Conv2d::forward(Tensor& x) {
    if (train) {
        this->input = &x;
    }
    int n = x.shape->n;
    int h = x.shape->h;
    int w = x.shape->w;
    Tensor& col = im2col(x);
    if (train) {
        this->input_col = &col;
    }
    this->filters->reshape(1,1, this->in_c * this->k * this->k, this->out_c);

    Tensor& out = filters->matmul(x);
    out.reshape(n, out_c, out_h, out_w);
    return out;
}

Tensor& Conv2d::sanity_check(Tensor& x) {
    this->out_h = (x.shape->h - this->k) / this->stride + 1;
    this->out_w = (x.shape->w - this->k) / this->stride + 1;
    return forward(x);
}

Tensor& Conv2d::backward(Tensor& grad) {
    Tensor& delta_w = grad.matmul(*input_col, nullptr, CblasNoTrans, CblasTrans);
    Tensor& col_grad = filters->matmul(grad, nullptr, CblasTrans, CblasNoTrans);

    Tensor& input_grad = col2im(col_grad, *(input->shape));

    return input_grad;
}
