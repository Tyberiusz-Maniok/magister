#include "conv2d.h"
#include "mkl.h"
#include <omp.h>
#include "consts.h"

using namespace lamp;

Conv2d::Conv2d(int input, int output, int kernel, int stride, Activation& activation_fn) : in_c(input), out_c(output),
    activation_fn(activation_fn), kernel(kernel), stride(stride) {
    this->filters = Tensor::random(new Shape(1, 1, input * kernel * kernel, output));
    this->bias = nullptr;
}

Conv2d::~Conv2d() {
    delete this->filters;
    delete this->bias;
    delete this->input_col;
}

void Conv2d::init_bias(Shape* shape) {
    this->bias = Tensor::random(shape);
}

Tensor& Conv2d::im2col(Tensor& x) {
    float* col_data = (float*) mkl_calloc(x.shape->n * out_h * out_w * x.shape->c * kernel * kernel, sizeof(float), MALLOC_ALIGN);
    Shape* col_shape = new Shape(x.shape->n, x.shape->c, kernel*kernel, out_h*out_w);
    Tensor* col = new Tensor(col_data, col_shape);

    #pragma omp parallel for simd collapse(6)
    for (int n = 0; n < x.shape->n; n++) {
        for (int c = 0; c < x.shape->c; c++) {
            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    for (int kh = 0; kh < kernel; kh++) {
                        for (int kw = 0; kw < kernel; kw++) {
                            *(col_data+col->flat_index(n, c, kh * kernel + kw, h * out_w + w)) = x.at(n, c, h * stride + kh, w * stride + kw);
                            // int col_idx = ((n * x.shape->c + c) * out_h + (h * stride) + kh) *  out_w + w * stride + kw;
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
            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    for (int kh = 0; kh < kernel; kh++) {
                        for (int kw = 0; kw < kernel; kw++) {
                            *(im_data+im->flat_index(n, c, h * stride + kh, w * stride + kw)) += x.at(n, c, kh * kernel + kw, h * out_w + w);
                            // int col_idx = ((n * x.shape->c + c) * out_h + (h * stride) + kh) *  out_w + w * stride + kw;
                            // *(im_data+im->flat_index(n, c, h * stride + kh, w * stride + kw)) = x.at(n, c, kh * kernel + kw, h * out_w + w);
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
        input_col = &col;
    }

    col.reshape(1, 1, col.shape->w * col.shape->n, col.shape->c * col.shape->h);
    Tensor& out = col.matmul(*filters);
    out.reshape(n, out_c, out_h, out_w);
    activation_fn.forward(out);
    return out;
}

Tensor& Conv2d::sanity_check(Tensor& x) {
    this->out_h = (x.shape->h - kernel) / stride + 1;
    this->out_w = (x.shape->w - kernel) / stride + 1;
    // this->bias = Tensor::random(new Shape(x.shape->n, out_c, out_h, out_w), rng);
    return forward(x);
}

Tensor& Conv2d::backward(Tensor& grad, float lr) {
    activation_fn.backward(grad);
    grad.reshape(1, 1, grad.shape->n * grad.shape->w, grad.shape->c * grad.shape->h);
    Tensor& delta_w = grad.matmul(*input_col);
    Tensor& col_grad = filters->matmul(grad);
    col_grad.reshape(input_col->shape->n, input_col->shape->c, input_col->shape->h, input_col->shape->w);

    filters->mulsub(delta_w, lr);
    // bias->mulsub(grad, lr);

    Tensor& input_grad = col2im(col_grad, *(input->shape));

    return input_grad;
}
