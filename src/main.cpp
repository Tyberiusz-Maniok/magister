#include <omp.h>
#include <cstdio>
#include <mkl.h>
#include "rng.h"
#include "tensor.h"

using namespace lamp;

int main() {

    // omp_set_num_threads(4);
    // #pragma omp parallel for
    // for (int i =0; i < 100; i++) {
    //     printf("%i", omp_get_thread_num());
    // }

    // int m = 300;
    // int k = 200;
    // int n = 400;

    // double *a, *b, *c;
    // a = (double*) mkl_malloc(m * k * sizeof(double), 64);
    // b = (double*) mkl_malloc(k * n * sizeof(double), 64);
    // c = (double*) mkl_malloc(m * n * sizeof(double), 64);

    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, k, b, n, 0, c, n);

    // mkl_free(a);
    // mkl_free(b);
    // mkl_free(c);

    RandomGen* r = new RandomGen(111);


    int* size1 = (int*) mkl_malloc(2* 4, 64);
    *size1 = 3;
    *(size1+1) = 4;

    int* size2 = (int*) mkl_malloc(2*4, 64);
    *size2 = 4;
    *(size2+1) = 5;
    
    // Tensor* a = new Tensor(d, size1, 2);
    Tensor* a = Tensor::random(size1, 2, *r);
    // Tensor* a = Tensor::zeros(size1, 2);
    Tensor* b = Tensor::random(size2, 2, *r);





    Tensor& c = a->matmul(*b);
    
    // mkl_free(d);
    delete r;
    delete a;
    delete b;
    delete &c;
    return 0;
}
