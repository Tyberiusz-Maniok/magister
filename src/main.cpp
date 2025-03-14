#include <omp.h>
#include <cstdio>
#include <mkl.h>

int main() {

    // omp_set_num_threads(4);
    // #pragma omp parallel for
    // for (int i =0; i < 100; i++) {
    //     printf("%i", omp_get_thread_num());
    // }

    int m = 300;
    int k = 200;
    int n = 400;

    double *a, *b, *c;
    a = (double*) mkl_malloc(m * k * sizeof(double), 64);
    b = (double*) mkl_malloc(k * n * sizeof(double), 64);
    c = (double*) mkl_malloc(m * n * sizeof(double), 64);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, k, b, n, 0, c, n);

    mkl_free(a);
    mkl_free(b);
    mkl_free(c);

    return 0;
}
