#include "conv2d.h"
#include "mkl.h"
#include <omp.h>
#include "consts.h"

using namespace lamp;

Conv2d::Conv2d(int input, int output, int kernel, int stride, Activation& activation_fn) : in_c(input), out_c(output),
    activation_fn(activation_fn), kernel(kernel), stride(stride) {
    this->filters = Tensor::random(new Shape(1, 1, output, input * kernel * kernel));
    this->bias = nullptr;
}

Conv2d::~Conv2d() {
}

void Conv2d::init_bias(Shape* shape) {
    this->bias = Tensor::random(shape);
}

TensorP Conv2d::im2col(TensorP x) {
    float* col_data = (float*) mkl_calloc(x->shape->n * out_h * out_w * x->shape->c * kernel * kernel, sizeof(float), MALLOC_ALIGN);
    Shape* col_shape = new Shape(x->shape->n, x->shape->c, kernel*kernel, out_h*out_w);
    TensorP col = TensorP(new Tensor(col_data, col_shape));

    #pragma omp parallel for simd collapse(6)
    for (int n = 0; n < x->shape->n; n++) {
        for (int c = 0; c < x->shape->c; c++) {
            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    for (int kh = 0; kh < kernel; kh++) {
                        for (int kw = 0; kw < kernel; kw++) {
                            *(col_data+col->flat_index(n, c, kh * kernel + kw, h * out_w + w)) = x->at(n, c, h * stride + kh, w * stride + kw);
                        }
                    }
                }
            }
        }
    }

    return col;
}

TensorP Conv2d::col2im(TensorP x, Shape* shape) {
    float* im_data = (float*) mkl_calloc(shape->c * shape->h * shape->w, sizeof(float), MALLOC_ALIGN);
    TensorP im = TensorP(new Tensor(im_data, new Shape(1, shape->c, shape->h, shape->w)));
    
    #pragma omp parallel for simd collapse(6)
    for (int n = 0; n < x->shape->n; n++) {
        for (int c = 0; c < in_c; c++) {
            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    for (int kh = 0; kh < kernel; kh++) {
                        for (int kw = 0; kw < kernel; kw++) {
                            *(im_data+im->flat_index(n, c, h * stride + kh, w * stride + kw)) += x->at(n, c, kh * kernel + kw, h * out_w + w);
                        }
                    }
                }
            }
        }
    }
    return im;
}

TensorP Conv2d::forward(TensorP x) {
    if (train) {
        this->input = x;
    }

    int n = x->shape->n;
    int h = x->shape->h;
    int w = x->shape->w;
    TensorP col = im2col(x);
    if (train) {
        input_col = col;
    }

    col->reshape(col->shape->n, 1, col->shape->c * col->shape->h, col->shape->w);
    TensorP out = filters->batched_matmul(col, this->bias);
    out->reshape(n, out_c, out_h, out_w);
    activation_fn.forward(out);

    return out;
}

TensorP Conv2d::sanity_check(TensorP x) {
    this->out_h = (x->shape->h - kernel) / stride + 1;
    this->out_w = (x->shape->w - kernel) / stride + 1;
    init_bias(new Shape(1, out_c, out_h, out_w));
    return forward(x);
}

TensorP Conv2d::backward(TensorP grad, float lr) {
    TensorP agrad = grad->avg_grad();
    TensorP ainput_col = this->input_col->avg_grad();
    activation_fn.backward(agrad);

    agrad->reshape(1, 1, agrad->shape->c * agrad->shape->h, agrad->shape->w);
    TensorP delta_w = agrad->matmul(ainput_col, nullptr, CblasNoTrans, CblasTrans);
    TensorP col_grad = filters->matmul(agrad, nullptr, CblasTrans);
    col_grad->reshape(ainput_col->shape->n, ainput_col->shape->c, ainput_col->shape->h, ainput_col->shape->w);

    filters->mulsub(delta_w, lr);
    bias->mulsub(agrad, lr);

    TensorP input_grad = col2im(col_grad, input->shape);

    return input_grad;
}
