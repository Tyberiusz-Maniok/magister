#include "conv2d.h"
#include <cstring>
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
    float* col_data = (float*) aligned_alloc(MALLOC_ALIGN, x->shape->n * out_h * out_w * x->shape->c * kernel * kernel * sizeof(float));
    Shape* col_shape = new Shape(x->shape->n, x->shape->c, kernel*kernel, out_h*out_w);
    TensorP col = TensorP(new Tensor(col_data, col_shape));

    // Extract values for GPU kernel (NCHW format)
    int x_n = x->shape->n;
    int x_c = x->shape->c;
    int x_stride_n = x->strides->n;
    int x_stride_c = x->strides->c;
    int x_stride_h = x->strides->h;
    int x_stride_w = x->strides->w;
    
    int col_stride_n = col->strides->n;
    int col_stride_c = col->strides->c;
    int col_stride_h = col->strides->h;
    int col_stride_w = col->strides->w;
    
    float* x_dev = x->d_data;
    float* col_dev = col->d_data;
    
    #pragma omp target teams distribute parallel for simd collapse(6) is_device_ptr(x_dev, col_dev)
    for (int n = 0; n < x_n; n++) {
        for (int c = 0; c < x_c; c++) {
            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    for (int kh = 0; kh < kernel; kh++) {
                        for (int kw = 0; kw < kernel; kw++) {
                            // Inline index calculations (NCHW format)
                            int col_idx = n * col_stride_n + c * col_stride_c + (kh * kernel + kw) * col_stride_h + (h * out_w + w) * col_stride_w;
                            int x_idx = n * x_stride_n + c * x_stride_c + (h * stride + kh) * x_stride_h + (w * stride + kw) * x_stride_w;
                            *(col_dev+col_idx) = *(x_dev+x_idx);
                        }
                    }
                }
            }
        }
    }

    return col;
}

TensorP Conv2d::col2im(TensorP x, Shape* shape) {
    float* im_data = (float*) aligned_alloc(MALLOC_ALIGN, shape->c * shape->h * shape->w * sizeof(float));
    memset(im_data, 0, shape->c * shape->h * shape->w * sizeof(float));
    TensorP im = TensorP(new Tensor(im_data, new Shape(1, shape->c, shape->h, shape->w)));

    // Extract stride values for GPU kernel (NCHW format)
    int x_stride_n = x->strides->n;
    int x_stride_c = x->strides->c;
    int x_stride_h = x->strides->h;
    int x_stride_w = x->strides->w;
    
    int im_stride_n = im->strides->n;
    int im_stride_c = im->strides->c;
    int im_stride_h = im->strides->h;
    int im_stride_w = im->strides->w;
    
    float* x_dev = x->d_data;
    float* im_dev = im->d_data;
    
    #pragma omp target teams distribute parallel for collapse(5) is_device_ptr(x_dev, im_dev)
    for (int c = 0; c < in_c; c++) {
        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                for (int kh = 0; kh < kernel; kh++) {
                    for (int kw = 0; kw < kernel; kw++) {
                        // Inline index calculations (NCHW format)
                        int im_idx = 0 * im_stride_n + c * im_stride_c + (h * stride + kh) * im_stride_h + (w * stride + kw) * im_stride_w;
                        int x_idx = 0 * x_stride_n + c * x_stride_c + (kh * kernel + kw) * x_stride_h + (h * out_w + w) * x_stride_w;
                        #pragma omp atomic
                        *(im_dev+im_idx) += *(x_dev+x_idx);
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
        this->input_col = col;
    }

    int col_c = col->shape->c;
    int col_h = col->shape->h;
    col->reshape(col->shape->n, 1, col->shape->c * col->shape->h, col->shape->w);
    TensorP out = filters->batched_matmul(col, this->bias);
    col->reshape(col->shape->n, col_c, col_h, col->shape->w);
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
    // Shape* shp = new Shape(*(input->shape));
    TensorP agrad = grad->avg_grad();
    TensorP ainput_col = this->input_col->avg_grad();
    activation_fn.backward(agrad);

    int col_c = ainput_col->shape->c;
    int col_h = ainput_col->shape->h;

    ainput_col->reshape(1, 1, ainput_col->shape->c * ainput_col->shape->h, ainput_col->shape->w);
    agrad->reshape(1, 1, agrad->shape->c, agrad->shape->h * agrad->shape->w);
    TensorP delta_w = agrad->matmul(ainput_col, nullptr, CUBLAS_OP_N, CUBLAS_OP_T);
    TensorP col_grad = filters->matmul(agrad, nullptr, CUBLAS_OP_T);

    ainput_col->reshape(1, col_c, col_h, ainput_col->shape->w);
    col_grad->reshape(1, ainput_col->shape->c, ainput_col->shape->h, ainput_col->shape->w);

    filters->mulsub(delta_w, lr);
    bias->mulsub(agrad, lr);

    TensorP input_grad = col2im(col_grad, new Shape(*(input->shape)));
    // delete shp;

    return input_grad;
}
