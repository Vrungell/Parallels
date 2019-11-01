#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>
#include <cmath>

void randomizeMatrix(int** matrix, int size) {
   for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = rand();
        }
    }
}

void spmd(int**a, int**b, int**res, int size, int num_threads) {
    omp_set_num_threads(num_threads);

    double executingTimeWithOmp = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            res[i][j] = 0;
            for (int k = 0; k < size; ++k) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    executingTimeWithOmp = omp_get_wtime() - executingTimeWithOmp;

    std::cout << executingTimeWithOmp << std::endl;
}

void sch_static(int**a, int**b, int**res, int size, int num_threads){
    omp_set_num_threads(num_threads);
    double executingTimeWithOmp = omp_get_wtime();
    #pragma omp parallel
    {
    #pragma omp for schedule(static, 1)
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            for (size_t k = 0; k < size; k++) {
                res[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
    executingTimeWithOmp = omp_get_wtime() - executingTimeWithOmp;
    std::cout <<executingTimeWithOmp <<std::endl;
}

void sch_dynamic(int**a, int**b, int**res, int size, int num_threads){
    omp_set_num_threads(num_threads);
    double executingTimeWithOmp = omp_get_wtime();
    #pragma omp parallel
    {
    #pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                for (size_t k = 0; k < size; k++) {
                    res[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
    executingTimeWithOmp = omp_get_wtime() - executingTimeWithOmp;
    std::cout <<  executingTimeWithOmp <<std::endl;
}

void sch_guided(int**a, int**b, int**res, int size, int num_threads){
    omp_set_num_threads(num_threads);
    double executingTimeWithOmp = omp_get_wtime();
    #pragma omp parallel
    {
    #pragma omp for schedule(guided, 1)
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                for (size_t k = 0; k < size; k++) {
                    res[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
    executingTimeWithOmp = omp_get_wtime() - executingTimeWithOmp;
    std::cout << executingTimeWithOmp << std::endl;
}

void sch_runtime(int**a, int**b, int**res, int size, int num_threads){
    omp_set_num_threads(num_threads);
    int rowsPerThread = ceil(size * 1. / num_threads);
    double executingTimeWithOmp = omp_get_wtime();
    #pragma omp parallel
    {
        int threadNum = omp_get_thread_num();
        size_t rowBegin = threadNum * rowsPerThread;
        size_t rowEnd = rowBegin + rowsPerThread;

        for (size_t i = rowBegin; i < fmin(rowEnd, size); i++) {
            for (size_t j = 0; j < size; j++) {
                for (size_t k = 0; k < size; k++) {
                    res[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
    executingTimeWithOmp = omp_get_wtime() - executingTimeWithOmp;
    std::cout << executingTimeWithOmp <<  std::endl;
}

int main() {
    int size = 1000;
    int **a = new int*[size];
    int **b = new int*[size];
    int **result = new int*[size];
    for (int i = 0; i < size; ++i) {
        a[i] = new int[size];
        b[i] = new int[size];
        result[i] = new int[size];
    }

    randomizeMatrix(a, size);
    randomizeMatrix(b, size);

    int max_threads = omp_get_max_threads();

    for (int threads = 1; threads <= max_threads; ++threads) {
    std::cout<<"SSDGR"<<threads<<std::endl;
    spmd(a,b,result,size,threads);
    sch_static(a,b,result,size,threads);
    sch_dynamic(a,b,result,size,threads);
    sch_guided(a,b,result,size,threads);
    sch_runtime(a,b,result,size,threads);
    }

    return 0;
}