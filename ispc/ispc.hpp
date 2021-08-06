
#include <device_launch_parameters.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "SGEMM_kernels_ispc.h"
#include "matrix.hpp"


bool tasks;

using namespace std;
using namespace ispc;


typedef void (*SGEMMFuncPtr)(void);
typedef void (*SGEMMFuncPtr_SingleThreaded_int)(int* matrixA, int* matrixB, int* matrixC, unsigned int A_ROW, unsigned int A_COL, unsigned int B_COL);
typedef void (*SGEMMFuncPtr_MultiThreaded_int)(int* matrixA, int* matrixB, int* matrixC, unsigned int A_ROW, unsigned int A_COL, unsigned int B_COL);
typedef void (*SGEMMFuncPtr_SingleThreaded_float)(float* matrixA, float* matrixB, float* matrixC, unsigned int A_ROW, unsigned int A_COL, unsigned int B_COL);
typedef void (*SGEMMFuncPtr_MultiThreaded_float)(float* matrixA, float* matrixB, float* matrixC, unsigned int A_ROW, unsigned int A_COL, unsigned int B_COL);
typedef void (*SGEMMFuncPtr_SingleThreaded_double)(double* matrixA, double* matrixB, double* matrixC, unsigned int A_ROW, unsigned int A_COL, unsigned int B_COL);
typedef void (*SGEMMFuncPtr_MultiThreaded_double)(double* matrixA, double* matrixB, double* matrixC, unsigned int A_ROW, unsigned int A_COL, unsigned int B_COL);

template <typename T1>
void Test_SGEMM(SGEMMFuncPtr SGEMMFunc, T1* matrixA, T1* matrixB, T1* matrixC,
    unsigned int A_ROW, unsigned int A_COL, unsigned int B_COL, bool task) {
    unsigned int i;
    if (task) {
        // type cast 
#if defined(USE_ISPC_INT)
        auto SGEMMFunc_MT = (SGEMMFuncPtr_MultiThreaded_int)SGEMMFunc;
#elif defined(USE_ISPC_FLOAT)
        auto SGEMMFunc_MT = (SGEMMFuncPtr_MultiThreaded_float)SGEMMFunc;
#elif defined(USE_ISPC_DOUBLE)
        auto SGEMMFunc_MT = (SGEMMFuncPtr_MultiThreaded_double)SGEMMFunc;
#endif
        for (i = 0; i < N; i++) {
            SGEMMFunc_MT(matrixA, matrixB, matrixC, A_ROW, A_COL, B_COL);
        }
    }

    else {
        // type cast
#if defined(USE_ISPC_INT)
        auto SGEMMFunc_ST = (SGEMMFuncPtr_SingleThreaded_int)SGEMMFunc;
#elif defined(USE_ISPC_FLOAT)
        auto SGEMMFunc_ST = (SGEMMFuncPtr_SingleThreaded_float)SGEMMFunc;
#elif defined(USE_ISPC_DOUBLE)
        auto SGEMMFunc_ST = (SGEMMFuncPtr_SingleThreaded_double)SGEMMFunc;
#endif
        for (i = 0; i < N; i++)
        {
            SGEMMFunc_ST(matrixA, matrixB, matrixC, A_ROW, A_COL, B_COL);
        }
    }
}

template <typename T1, typename T2>
void cpu_ispc() {

    int programCount = SGEMM_get_program_count();
    int tileSize = SGEMM_get_tile_size();
    if (B_COL % programCount != 0 || B_COL % tileSize != 0) {
        printf("\nNumber of columns in Matrix B (K) must be a multiple of %d (target width) and %d (tile size)!\n",
            programCount, tileSize);
        exit(-1);
    }

    if (A_ROW % programCount != 0) {
        printf("\nNumber of rows in Matrix A (M) must be a multiple of %d (target width)!\n", programCount);
        exit(-1);
    }

    if (A_COL % programCount != 0) {
        printf("\nNumber of columns in Matrix A (N), which is also number of rows in Matrix B, "
            "must be a multiple of %d (target width)!\n",
            programCount);
        exit(-1);
    }


    T1* h_A, * h_B;
    T2* h_C, * h_CC;

    
    h_A = (T1*)malloc(sizeof(T1) * A_ROW * A_COL);
    h_B = (T1*)malloc(sizeof(T1) * B_ROW * B_COL);
    h_C = (T2*)malloc(sizeof(T2) * A_ROW * B_COL);
    h_CC = (T2*)malloc(sizeof(T2) * A_ROW * B_COL);
    
        /*
    cudaHostAlloc((void**)&h_A, sizeof(T1) * A_ROW * A_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_B, sizeof(T1) * B_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_C, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_CC, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);
    */

    //初始化矩阵
    MatrixINIT<T1>(A_ROW, A_COL, h_A);
    MatrixINIT<T1>(B_ROW, B_COL, h_B);
   
    //打印矩阵
    //Matrixshow<T1>("A", A_ROW, A_COL, h_A, 1);
    //Matrixshow<T1>("B", B_ROW, B_COL, h_B, 1);

    // Single threaded test cases:
    tasks = false;
    TIMER_START(t);
#if defined(USE_ISPC_INT)
    Test_SGEMM<T1>((SGEMMFuncPtr)SGEMM_tileBlockNoSIMDIntrin_2_int, h_A, h_B,
        h_C, A_ROW, A_COL, B_COL, tasks);
#elif defined(USE_ISPC_FLOAT)
    Test_SGEMM<T1>((SGEMMFuncPtr)SGEMM_tileBlockNoSIMDIntrin_2_float, h_A, h_B,
        h_C, A_ROW, A_COL, B_COL, tasks);
#elif defined(USE_ISPC_DOUBLE)
    Test_SGEMM<T1>((SGEMMFuncPtr)SGEMM_tileBlockNoSIMDIntrin_2_double, h_A, h_B,
        h_C, A_ROW, A_COL, B_COL, tasks);
#endif
    TIMER_STOP(t);
    cout << "ISPC单任务花费了：" << TIMER_MSEC(t) << " ms " << endl << endl;

    cout << endl;

    //打印结果
    //Matrixshow<T2>("ISPC 计算结果C的值:", A_ROW, B_COL, h_C, 1,0);
#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C, h_CC, 0);
#endif 

    // Multi-threaded test cases:
    tasks = true;
    TIMER_START(X);
#if defined(USE_ISPC_INT)
    Test_SGEMM<T1>((SGEMMFuncPtr)SGEMM_tileBlockNoSIMDIntrin_2_withTasks_int,
        h_A, h_B, h_C, A_ROW, A_COL, B_COL, tasks);
#elif defined(USE_ISPC_FLOAT)
    Test_SGEMM<T1>((SGEMMFuncPtr)SGEMM_tileBlockNoSIMDIntrin_2_withTasks_float,
        h_A, h_B, h_C, A_ROW, A_COL, B_COL, tasks);
#elif defined(USE_ISPC_DOUBLE)
    Test_SGEMM<T1>((SGEMMFuncPtr)SGEMM_tileBlockNoSIMDIntrin_2_withTasks_double,
        h_A, h_B, h_C, A_ROW, A_COL, B_COL, tasks);
#endif
    TIMER_STOP(X);

    cout << "ISPC多任务花费了：" << TIMER_MSEC(X) << " ms " << endl << endl;
    cout << endl;

    //打印结果
    //Matrixshow<T2>("ISPC 计算结果C的值:", A_ROW, B_COL, h_C, 1, 0);

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C, h_CC, 0);
#endif 

    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_CC);
    /*
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_CC);
    */
}
