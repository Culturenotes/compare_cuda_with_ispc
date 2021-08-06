
#include <device_launch_parameters.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>
#include <memory>
#include "ispc/ispc.hpp"

using namespace std;

#define BLOCK_SIZE 32

float elapsedTime_mykernel, elapsedTime_cublas, elapsedTime;
cudaEvent_t start, stop; 

__global__ void warm_up()
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("----------------warm_up---------------- %d \n",tid);
}


template <typename T1, typename T2>
__global__  void MatrixMulCUDA(const T1* A, const T1 * B, T2* C,
     const int ROW_A, const int COL_A, const int ROW_B, const int COL_B)
 {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    //当前线程对应的矩阵C的元素位置
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;
    //int I = (COL_A + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int I = (COL_A + BLOCK_SIZE - 1) / BLOCK_SIZE;

    T2 t=0.0f, Csub = 0.0f, comp = 0.0f;
 
    __shared__ float As[BLOCK_SIZE+1][BLOCK_SIZE+1];
    __shared__ float Bs[BLOCK_SIZE+1][BLOCK_SIZE+1];

    //每个Block都将遍历A的一整行块和B的一整列块
    //每个线程主要负责一行和一列的内积，另外还负责为当前循环中所计算的块填充一个元素到共享内存中
    //快速向上取整
    for (int i = 0; i < I; i++) {
        
        if (row < ROW_A && i * BLOCK_SIZE + tx < COL_A)
            As[ty][tx] = A[row * COL_A + i * BLOCK_SIZE + tx];//所有计算单元同时加载，所以下面的for循环中As和Bs都已配置完成
        else
            As[ty][tx] = 0;

        if (col < COL_B && i * BLOCK_SIZE + ty < ROW_B)
            Bs[ty][tx] = B[(i * BLOCK_SIZE + ty) * COL_B + col];
        else
            Bs[ty][tx] = 0;

        //让同一块中的不同线程指令流同步，保证共享内存中矩阵块的元素全部加载
        __syncthreads();//各线程执行到此函数时等待，直到全部线程同步


        //Kahan's Summation Formula
        //虽然外层循环是面向Block的，但这里内层循环只计算了两块中某行和某列的
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
            //  c += As[ty][j] * Bs[j][tx];
            comp -= As[ty][j] * Bs[j][tx];
            t = Csub - comp;
            comp = (t - Csub) + comp;
            Csub = t;
        }
          
        __syncthreads();
    }
    if (row < ROW_A && col < COL_B)
    {
        C[row * COL_B + col] = Csub;
    }
 }


template <typename T1, typename T2>
__global__ void  MatrixMulCUDA_2D(
    const T1* A, size_t pitchA, const T1* B, size_t pitchB, T2* C, size_t pitchC,
    const int ROW_A, const int COL_A, const int ROW_B, const int COL_B)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //当前线程对应的矩阵C的元素位置
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    int I = (COL_A + BLOCK_SIZE - 1) / BLOCK_SIZE;

    T2 t = 0.0f, Csub = 0.0f, comp = 0.0f;

    __shared__ float AS[BLOCK_SIZE+1][BLOCK_SIZE+1];
    __shared__ float BS[BLOCK_SIZE+1][BLOCK_SIZE+1];

    for (int i = 0; i < I; i++)
    {

        if (row < ROW_A && i * BLOCK_SIZE + tx < COL_A)
        {
            AS[ty][tx] = A[row * pitchA + i * BLOCK_SIZE + tx];
        }
        else
        {
            AS[ty][tx] = 0;
        }
        if (col < COL_B && i * BLOCK_SIZE + ty < ROW_B)
        {
            BS[ty][tx] = B[(i * BLOCK_SIZE + ty) * pitchB + col];
        }
        else
        {
            BS[ty][tx] = 0;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            comp -= AS[ty][k] * BS[k][tx];
            t = Csub - comp;
            comp = (t - Csub) + comp;
            Csub = t;
        }
        __syncthreads();
    }

    if (row < ROW_A && col < COL_B)
    {
        C[row * pitchC + col] = Csub;
       // C[(by * BLOCK_SIZE + ty) * pitchC + bx * BLOCK_SIZE + tx] = Csub;
    }
}


template <typename T1, typename T2>
void My_kernel()
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //分配CPU上的存储空间
    T1* h_a, * h_b, * h_c, * h_cc;
    cudaHostAlloc((void**)&h_a, sizeof(T1) * A_ROW * A_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_b, sizeof(T1) * B_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_c, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_cc, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);

    MatrixINIT<T1>(A_ROW, A_COL, h_a);
    MatrixINIT<T1>(B_ROW, B_COL, h_b);

    /*
    Matrixshow<T1>("A", A_ROW, A_COL, h_a, 0);
    Matrixshow<T1>("B", B_ROW, B_COL, h_b, 0);
    */

    //分配GPU上的存储空间
    T1* d_a, * d_b;
    T2* d_c;

    /*
    cudaHostAlloc((void**)&d_a, sizeof(T1) * A_ROW * A_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&d_b, sizeof(T1) * B_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&d_c, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);
    */
    
    cudaMalloc((void**)&d_a, sizeof(T1) * A_ROW * A_COL);
    cudaMalloc((void**)&d_b, sizeof(T1) * B_ROW * B_COL);
    cudaMalloc((void**)&d_c, sizeof(T2) * A_ROW * B_COL);

    unsigned int grid_rows = (A_ROW + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (B_COL + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(grid_rows, grid_cols);
    dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);

    //创建流对象，用于任务级(Grid)同步
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    //计时开始
    TIMER_START(_X);
    for (int i = 0; i < N; ++i)
    {
        // copy matrix A and B from host to device memory
        cudaMemcpyAsync(d_a, h_a, sizeof(T1) * A_ROW * A_COL, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_b, h_b, sizeof(T1) * B_ROW * B_COL, cudaMemcpyHostToDevice, stream);

        //cudaMemcpy(d_a, h_a, sizeof(T1) * A_ROW * A_COL, cudaMemcpyHostToDevice);
        //cudaMemcpy(d_b, h_b, sizeof(T1) * B_ROW * B_COL, cudaMemcpyHostToDevice);
        

        cudaEventRecord(start, 0);
        
        MatrixMulCUDA<T1, T2> << < grid, blocks >> > (d_a, d_b, d_c, A_ROW, A_COL, B_ROW, B_COL);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime_mykernel, start, stop);

        cudaMemcpyAsync(h_c, d_c, sizeof(T2) * A_ROW * B_COL, cudaMemcpyDeviceToHost, stream);
        
        //cudaMemcpy(h_c, d_c, sizeof(T2) * A_ROW * B_COL, cudaMemcpyDeviceToHost);
        
    }

    TIMER_STOP(_X);
    cout <<"mykernel GPU传输、计算花费了: " << TIMER_MSEC(_X) << " ms " << "\n";

    std::cout <<"mykernel GPU计算花费了："<<elapsedTime_mykernel * N<< " ms" << std::endl;
    //Matrixshow<T2>("计算结果矩阵C的值", A_ROW, B_COL, h_c, 0);
    cout << endl;

    //检查计算是否正确
#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_a, h_b, A_ROW, A_COL, B_COL, h_c, h_cc, 0);
#endif 

    //清理内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_b);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}

template <typename T1, typename T2>
//传入必须是方正才行
void My_kernel_2D()
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    T1* h_a, * h_b, * h_c, * h_cc;
    cudaHostAlloc((void**)&h_a, sizeof(T1) * A_ROW * A_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_b, sizeof(T1) * B_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_c, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_cc, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);

    MatrixINIT<T1>(A_ROW, A_COL, h_a);
    MatrixINIT<T1>(B_ROW, B_COL, h_b);

    Matrixshow<T1>("A", A_ROW, A_COL, h_a, 0);
    Matrixshow<T1>("B", B_ROW, B_COL, h_b, 0);

    T1 *d_a, *d_b;
    T2 *d_c;
    size_t pitch_a, pitch_b, pitch_c;

    //cudaMalloc((void**)&d_a, sizeof(T1) * A_ROW * A_COL);
    //cudaMalloc((void**)&d_b, sizeof(T1) * B_ROW * B_COL);
    //cudaMalloc((void**)&d_c, sizeof(T2) * A_ROW * B_COL);

    /*
    cudaHostAlloc((void**)&d_a, sizeof(T1) * A_ROW * A_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&d_b, sizeof(T1) * B_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&d_c, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);
    */

    cudaMallocPitch((void**)&d_a, &pitch_a, sizeof(T1) * A_ROW, A_COL);
    cudaMallocPitch((void**)&d_b, &pitch_b, sizeof(T1) * B_ROW, B_COL);
    cudaMallocPitch((void**)&d_c, &pitch_c, sizeof(T2) * A_ROW, B_COL);

    //创建流对象，用于任务级(Grid)同步
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
     
    unsigned int grid_rows = (A_ROW + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (B_COL + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(grid_rows, grid_cols);
    dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);

    //计时开始
    TIMER_START(_X);
    //cudaEventRecord(start, 0);

    for (int i = 0; i < N; ++i)
    {   
        // copy matrix A and B from host to device memory
         /*
        cudaMemcpy2D(d_a, pitch_a, h_a, sizeof(T1) * A_ROW, sizeof(T1) * A_ROW, A_COL, cudaMemcpyHostToDevice);
        cudaMemcpy2D(d_b, pitch_b, h_b, sizeof(T1) * B_ROW, sizeof(T1) * B_ROW, B_COL, cudaMemcpyHostToDevice);
         */
        cudaMemcpy2DAsync(d_a, pitch_a, h_a, sizeof(T1) * A_ROW, sizeof(T1) * A_ROW, A_COL,
            cudaMemcpyHostToDevice, stream);
        cudaMemcpy2DAsync(d_b, pitch_b, h_b, sizeof(T1) * B_ROW, sizeof(T1) * B_ROW, B_COL,
            cudaMemcpyHostToDevice, stream);

        //MatMultiSharePitch_Kernel<T1, T2> << <grid, blocks >> > (d_a, d_b, d_c, pitch_a / sizeof(T1), pitch_b / sizeof(T1), pitch_c / sizeof(T2));
        //gpu_matrix_mult_2D<T1,T2> <<<grid, blocks, sizeof(T2)* A_ROW >> > (d_a, pitch_a / sizeof(T1), d_b, pitch_b / sizeof(T1), d_c, pitch_c / sizeof(T2), A_ROW,A_COL,B_ROW, B_COL);
        //matMultCUDA_2D<T1,T2><<<grid, blocks>>> (d_a, pitch_a / sizeof(T1), d_b, pitch_b / sizeof(T1), d_c, pitch_c / sizeof(T2), A_ROW, A_COL, B_ROW, B_COL);
        MatrixMulCUDA_2D< T1, T2 > << <grid, blocks>> > (d_a, pitch_a / sizeof(T1), d_b, pitch_b / sizeof(T1), d_c, pitch_c / sizeof(T2), A_ROW, A_COL, B_ROW, B_COL);


        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime_mykernel, start, stop);

        cudaMemcpy2DAsync(h_c, sizeof(T2) * A_ROW, d_c, pitch_c, sizeof(T2) * A_ROW, B_COL, cudaMemcpyDeviceToHost, stream);
        //cudaMemcpy2D(h_c, sizeof(T2) * A_ROW, d_c, pitch_c, sizeof(T2) * A_ROW, B_COL, cudaMemcpyDeviceToHost);
    }
  
    //计时结束
    TIMER_STOP(_X);
    cout << "my_kernel_2D GPU传输、计算花费了: " << TIMER_MSEC(_X) << " ms " << "\n";
    std::cout<< "my_kernel_2D GPU花费了：" << elapsedTime_mykernel * N << " ms " << std::endl;

    //Matrixshow<T2>("计算结果矩阵C的值", A_ROW, B_COL, h_c, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_a, h_b, A_ROW, A_COL, B_COL, h_c, h_cc, 0);
#endif 

    //清理内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_b);
    /*
    cudaFreeHost(d_a);
    cudaFreeHost(d_b);
    cudaFreeHost(d_b);
    */
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}


template <typename T1, typename T2>
void cublas_kernel_asny()
{
    // 定义状态变量
    cublasHandle_t handle[2];
    for (int i = 0; i < 2; i++)
    {
        cublasCreate(&handle[i]);
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //存储于内存中的矩阵
    T1* h_A, * h_B;
    T2* h_C0, * h_C1, * h_CC;

    //在内存中开辟空间
    cudaHostAlloc((void**)&h_A, sizeof(T1) * A_ROW * A_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_B, sizeof(T1) * B_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_C0, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_C1, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_CC, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);



    MatrixINIT<T1>(A_ROW, A_COL, h_A);
    MatrixINIT<T1>(B_ROW, B_COL, h_B);

    // 打印待测试的矩阵
#if defined(USE_FLOAT_T)
    Matrixshow<T1>("A", A_ROW, A_COL, h_A, 0);
    Matrixshow<T1>("B", B_ROW, B_COL, h_B, 0);

#elif defined(USE_FLOAT_N)
    Matrixshow<T1>("A", A_ROW, A_COL, h_A, 0);
    Matrixshow<T1>("B", B_ROW, B_COL, h_B, 0);

#elif defined(USE_DOUBLE_T)
    Matrixshow<T1>("A", A_ROW, A_COL, h_A, 0);
    Matrixshow<T1>("B", B_ROW, B_COL, h_B, 0);

#elif defined(USE_DOUBLE_N)
    Matrixshow<T1>("A", A_ROW, A_COL, h_A, 0);
    Matrixshow<T1>("B", B_ROW, B_COL, h_B, 0);


#elif defined(USE_INT8_T)
    Matrixshow<T1>("A", A_ROW, A_COL, h_A, 0, 0, "char");
    Matrixshow<T1>("B", B_ROW, B_COL, h_B, 0, 0, "char");

#elif defined(USE_INT8_N)
    Matrixshow<T1>("A", A_ROW, A_COL, h_A, 0, 0, "char");
    Matrixshow<T1>("B", B_ROW, B_COL, h_B, 0, 0, "char");
#endif

    //存储于显存中的矩阵
    T1* d_A0, * d_A1, * d_B0, * d_B1;
    T2* d_C0, * d_C1;

    cudaMalloc((void**)&d_A0, sizeof(T1) * A_ROW * A_COL);
    cudaMalloc((void**)&d_B0, sizeof(T1) * B_ROW * B_COL);
    cudaMalloc((void**)&d_C0, sizeof(T2) * A_ROW * B_COL);

    cudaMalloc((void**)&d_A1, sizeof(T1) * A_ROW * A_COL);
    cudaMalloc((void**)&d_B1, sizeof(T1) * B_ROW * B_COL);
    cudaMalloc((void**)&d_C1, sizeof(T2) * A_ROW * B_COL);

    /*
    cudaHostAlloc((void**)&d_A0, sizeof(T1) * A_ROW * A_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&d_B0, sizeof(T1) * B_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&d_C0, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);

    cudaHostAlloc((void**)&d_A1, sizeof(T1) * A_ROW * A_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&d_B1, sizeof(T1) * B_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&d_C1, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);
    */

    //申明并创建流
    cudaStream_t stream[2];
    for (int i = 0; i < 2; i++)
    {
        cudaStreamCreate(&stream[i]);
    }

    //绑定cublas句柄和流
    for (int i = 0; i < 2; i++)
    {
        cublasSetStream(handle[i], stream[i]);
    }

    const T2 a = 1.0f, b = 0.0f;

    //数据从Host端拷贝到Device端、 传统方式
    //cudaMemcpy(d_A, H_A, sizeof(T1) * A_ROW * A_COL, cudaMemcpyHostToDevice); 
    //cudaMemcpy(d_B, H_B, sizeof(T1) * B_ROW * B_COL, cudaMemcpyHostToDevice);

     //计时开始
    TIMER_START(_X);
    //cudaEventRecord(start, 0);

    for (int i = 0; i < N / 2; i++)
    {
        /*
        //数据从Host端拷贝到Device端、 多流传输方式
        cudaMemcpyAsync(d_A0, h_A, sizeof(T1) * A_ROW * A_COL, cudaMemcpyHostToDevice, stream[0]);
        cudaMemcpyAsync(d_B0, h_B, sizeof(T1) * B_ROW * B_COL, cudaMemcpyHostToDevice, stream[0]);
        cudaMemcpyAsync(d_A1, h_A, sizeof(T1) * A_ROW * A_COL, cudaMemcpyHostToDevice, stream[1]);
        cudaMemcpyAsync(d_B1, h_B, sizeof(T1) * B_ROW * B_COL, cudaMemcpyHostToDevice, stream[1]);
        */

        
       //数据从Host端拷贝到Device端、 cubals方式
        /*
        cublasSetVectorAsync(A_ROW * A_COL, sizeof(T1), h_A, 1, d_A0, 1,stream[0]);
        cublasSetVectorAsync(B_ROW * B_COL, sizeof(T1), h_B, 1, d_B0, 1, stream[0]);
        cublasSetVectorAsync(A_ROW * A_COL, sizeof(T1), h_A, 1, d_A1, 1, stream[1]);
        cublasSetVectorAsync(B_ROW * B_COL, sizeof(T1), h_B, 1, d_B1, 1, stream[1]);
        */
        
        cublasSetMatrixAsync(A_ROW, A_COL, sizeof(*h_A), h_A, A_ROW, d_A0, A_ROW, stream[0]);
        cublasSetMatrixAsync(B_ROW, B_COL, sizeof(*h_B), h_B, B_ROW, d_B0, B_ROW, stream[0]);
        cublasSetMatrixAsync(A_ROW, A_COL, sizeof(*h_A), h_A, A_ROW, d_A1, A_ROW, stream[1]);
        cublasSetMatrixAsync(B_ROW, B_COL, sizeof(*h_B), h_B, B_ROW, d_B1, B_ROW, stream[1]);
        
        cudaEventRecord(start, 0);
#if defined(USE_FLOAT_N)    
        cublasSgemm(
            handle[0],
            CUBLAS_OP_N,   //矩阵A的属性参数，不转置，按列优先
            CUBLAS_OP_N,   //矩阵B的属性参数，不转置，按列优先
            B_COL,          //矩阵B^T、C^T的行数
            A_ROW,          //矩阵A^T、C^T的列数
            B_ROW,          //B^T的列数，A^T的行数，此处也可为A_COL,一样的
            &a,             //alpha的值
            d_B0,            //左矩阵，为B^T
            B_COL,          //B^T的leading dimension，按列优先，则leading dimension为B^T的行数(B的列数)
            d_A0,            //右矩阵，为A^T
            A_COL,          //A^T的leading dimension，按列优先，则leading dimension为A^T的行数(A的列数)
            &b,             //beta的值
            d_C0,            //结果矩阵C
            B_COL           //C^T的leading dimension，C^T矩阵一定按列优先，则leading dimension为C^T的行数(C的列数)
        );//此处的h_C是按列存储的C^T

        cublasSgemm(
            handle[1],
            CUBLAS_OP_N,   //矩阵A的属性参数，不转置，按列优先
            CUBLAS_OP_N,   //矩阵B的属性参数，不转置，按列优先
            B_COL,          //矩阵B^T、C^T的行数
            A_ROW,          //矩阵A^T、C^T的列数
            B_ROW,          //B^T的列数，A^T的行数，此处也可为A_COL,一样的
            &a,             //alpha的值
            d_B1,            //左矩阵，为B^T
            B_COL,          //B^T的leading dimension，按列优先，则leading dimension为B^T的行数(B的列数)
            d_A1,            //右矩阵，为A^T
            A_COL,          //A^T的leading dimension，按列优先，则leading dimension为A^T的行数(A的列数)
            &b,             //beta的值
            d_C1,            //结果矩阵C
            B_COL           //C^T的leading dimension，C^T矩阵一定按列优先，则leading dimension为C^T的行数(C的列数)
        );//此处的h_C是按列存储的C^T

#elif defined(USE_FLOAT_T)
        cublasSgemm(
            handle[0],
            CUBLAS_OP_T,   //矩阵A的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            CUBLAS_OP_T,   //矩阵B的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            A_ROW,          //矩阵A、C的行数
            B_COL,          //矩阵B、C的列数
            A_COL,          //A的列数，B的行数，此处也可为B_ROW一样的
            &a,             //alpha的值
            d_A0,            //左矩阵，为A
            A_COL,          //A的leading dimension，按列优先，则leading dimension为（A^T的行数）A的列数
            d_B0,            //右矩阵，为B
            B_COL,          //B的leading dimension，按列优先，则leading dimension为（B^T的行数）A的列数
            &b,             //beta的值
            d_C0,            //结果矩阵C
            A_ROW           //C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
        );//此处的h_C是按列存储的C

        cublasSgemm(
            handle[1],
            CUBLAS_OP_T,   //矩阵A的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            CUBLAS_OP_T,   //矩阵B的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            A_ROW,          //矩阵A、C的行数
            B_COL,          //矩阵B、C的列数
            A_COL,          //A的列数，B的行数，此处也可为B_ROW一样的
            &a,             //alpha的值
            d_A1,            //左矩阵，为A
            A_COL,          //A的leading dimension，按列优先，则leading dimension为（A^T的行数）A的列数
            d_B1,            //右矩阵，为B
            B_COL,          //B的leading dimension，按列优先，则leading dimension为（B^T的行数）A的列数
            &b,             //beta的值
            d_C1,            //结果矩阵C
            A_ROW           //C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
        );//此处的h_C是按列存储的C

#elif defined(USE_DOUBLE_T)
        cublasDgemm(
            handle[0],
            CUBLAS_OP_T,   //矩阵A的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            CUBLAS_OP_T,   //矩阵B的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            A_ROW,          //矩阵A、C的行数
            B_COL,          //矩阵B、C的列数
            A_COL,          //A的列数，B的行数，此处也可为B_ROW一样的
            &a,             //alpha的值
            d_A0,            //左矩阵，为A
            A_COL,          //A的leading dimension，按列优先，则leading dimension为（A^T的行数）A的列数
            d_B0,            //右矩阵，为B
            B_COL,          //B的leading dimension，按列优先，则leading dimension为（B^T的行数）A的列数
            &b,             //beta的值
            d_C0,            //结果矩阵C
            A_ROW           //C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
        );//此处的h_C是按列存储的C

        cublasDgemm(
            handle[1],
            CUBLAS_OP_T,   //矩阵A的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            CUBLAS_OP_T,   //矩阵B的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            A_ROW,          //矩阵A、C的行数
            B_COL,          //矩阵B、C的列数
            A_COL,          //A的列数，B的行数，此处也可为B_ROW一样的
            &a,             //alpha的值
            d_A1,            //左矩阵，为A
            A_COL,          //A的leading dimension，按列优先，则leading dimension为（A^T的行数）A的列数
            d_B1,            //右矩阵，为B
            B_COL,          //B的leading dimension，按列优先，则leading dimension为（B^T的行数）A的列数
            &b,             //beta的值
            d_C1,            //结果矩阵C
            A_ROW           //C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
        );//此处的h_C是按列存储的C

#elif defined(USE_DOUBLE_N)
        cublasDgemm(
            handle[0],
            CUBLAS_OP_N,   //矩阵A的属性参数，不转置，按列优先
            CUBLAS_OP_N,   //矩阵B的属性参数，不转置，按列优先
            B_COL,          //矩阵B^T、C^T的行数
            A_ROW,          //矩阵A^T、C^T的列数
            B_ROW,          //B^T的列数，A^T的行数，此处也可为A_COL,一样的
            &a,             //alpha的值
            d_B0,            //左矩阵，为B^T
            B_COL,          //B^T的leading dimension，按列优先，则leading dimension为B^T的行数(B的列数)
            d_A0,            //右矩阵，为A^T
            A_COL,          //A^T的leading dimension，按列优先，则leading dimension为A^T的行数(A的列数)
            &b,             //beta的值
            d_C0,            //结果矩阵C
            B_COL           //C^T的leading dimension，C^T矩阵一定按列优先，则leading dimension为C^T的行数(C的列数)
        );//此处的h_C是按列存储的C^T

        cublasDgemm(
            handle[1],
            CUBLAS_OP_N,   //矩阵A的属性参数，不转置，按列优先
            CUBLAS_OP_N,   //矩阵B的属性参数，不转置，按列优先
            B_COL,          //矩阵B^T、C^T的行数
            A_ROW,          //矩阵A^T、C^T的列数
            B_ROW,          //B^T的列数，A^T的行数，此处也可为A_COL,一样的
            &a,             //alpha的值
            d_B1,            //左矩阵，为B^T
            B_COL,          //B^T的leading dimension，按列优先，则leading dimension为B^T的行数(B的列数)
            d_A1,            //右矩阵，为A^T
            A_COL,          //A^T的leading dimension，按列优先，则leading dimension为A^T的行数(A的列数)
            &b,             //beta的值
            d_C1,            //结果矩阵C
            B_COL           //C^T的leading dimension，C^T矩阵一定按列优先，则leading dimension为C^T的行数(C的列数)
        );//此处的h_C是按列存储的C^T


#elif defined(USE_INT8_N)
        cublasGemmEx(handle[0],  //句柄
            CUBLAS_OP_N,      //矩阵A的属性参数，不转置，按列优先
            CUBLAS_OP_N,      //矩阵B的属性参数，不转置，按列优先
            B_COL,            //矩阵B^T、C^T的行数
            A_ROW,            //矩阵A^T、C^T的列数
            B_ROW,            //B^T的列数，A^T的行数，此处也可为A_COL,一样的
            &a,               //alpha的值
            d_B0,              //左矩阵，为B^T
            CUDA_R_8I,        //A矩阵计算模式，int8型
            B_COL,            //B^T的leading dimension，按列优先，则leading dimension为B^T的行数(B的列数)
            d_A0,              //右矩阵，为A^T
            CUDA_R_8I,        //B矩阵计算模式，int8型
            A_COL,            //A^T的leading dimension，按列优先，则leading dimension为A^T的行数(A的列数)
            &b,               //乘法因子beta
            d_C0,              //C结果矩阵
            CUDA_R_32I,       //C矩阵计算模式，int32型
            B_COL,             //C^T的leading dimension，C^T矩阵一定按列优先，则leading dimension为C^T的行数(C的列数)
            CUDA_R_32I,       //计算模式，int32模式
            //CUBLAS_GEMM_ALGO0    //算法参数
            CUBLAS_GEMM_DFALT
        );                    //此处的h_C是按列存储的C^T

        cublasGemmEx(handle[1],  //句柄
            CUBLAS_OP_N,      //矩阵A的属性参数，不转置，按列优先
            CUBLAS_OP_N,      //矩阵B的属性参数，不转置，按列优先
            B_COL,            //矩阵B^T、C^T的行数
            A_ROW,            //矩阵A^T、C^T的列数
            B_ROW,            //B^T的列数，A^T的行数，此处也可为A_COL,一样的
            &a,               //alpha的值
            d_B1,              //左矩阵，为B^T
            CUDA_R_8I,        //A矩阵计算模式，int8型
            B_COL,            //B^T的leading dimension，按列优先，则leading dimension为B^T的行数(B的列数)
            d_A1,              //右矩阵，为A^T
            CUDA_R_8I,        //B矩阵计算模式，int8型
            A_COL,            //A^T的leading dimension，按列优先，则leading dimension为A^T的行数(A的列数)
            &b,               //乘法因子beta
            d_C1,              //C结果矩阵
            CUDA_R_32I,       //C矩阵计算模式，int32型
            B_COL,             //C^T的leading dimension，C^T矩阵一定按列优先，则leading dimension为C^T的行数(C的列数)
            CUDA_R_32I,       //计算模式，int32模式
            //CUBLAS_GEMM_ALGO0    //算法参数
            CUBLAS_GEMM_DFALT
        );                    //此处的h_C是按列存储的C^T

#elif defined(USE_INT8_T)
        cublasGemmEx(handle[0],      //句柄
            CUBLAS_OP_T,          //矩阵A的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            CUBLAS_OP_T,          //矩阵B的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            A_ROW,                //矩阵A、C的行数
            B_COL,                //矩阵B、C的列数
            A_COL,                //A的列数，B的行数，此处也可为B_ROW一样的
            &a,                   //运算式的 α 值
            d_A0,                  //A矩阵
            CUDA_R_8I,            //A矩阵计算模式，int8型
            A_COL,                //A的leading dimension，按行优先存储，读取还是列优先，则leading dimension为（A^T的行数）A的列数
            d_B0,                  //B矩阵
            CUDA_R_8I,            //B矩阵计算模式，int8型
            B_COL,                //B的leading dimension，按行优先存储，读取还是列优先，则leading dimension为（B^T的行数）A的列数
            &b,                   //乘法因子beta
            d_C0,                  //C结果矩阵
            CUDA_R_32I,           //C矩阵计算模式，int32型
            A_ROW,                //C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
            CUDA_R_32I,           //计算模式，int32模式
            //CUBLAS_GEMM_ALGO2     //算法参数
            CUBLAS_GEMM_DFALT
        );                        //此处的h_C是按列存储的C


        cublasGemmEx(handle[1],      //句柄
            CUBLAS_OP_T,          //矩阵A的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            CUBLAS_OP_T,          //矩阵B的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            A_ROW,                //矩阵A、C的行数
            B_COL,                //矩阵B、C的列数
            A_COL,                //A的列数，B的行数，此处也可为B_ROW一样的
            &a,                   //运算式的 α 值
            d_A1,                  //A矩阵
            CUDA_R_8I,            //A矩阵计算模式，int8型
            A_COL,                //A的leading dimension，按行优先存储，读取还是列优先，则leading dimension为（A^T的行数）A的列数
            d_B1,                  //B矩阵
            CUDA_R_8I,            //B矩阵计算模式，int8型
            B_COL,                //B的leading dimension，按行优先存储，读取还是列优先，则leading dimension为（B^T的行数）A的列数
            &b,                   //乘法因子beta
            d_C1,                  //C结果矩阵
            CUDA_R_32I,           //C矩阵计算模式，int32型
            A_ROW,                //C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
            CUDA_R_32I,           //计算模式，int32模式
            //CUBLAS_GEMM_ALGO2     //算法参数
            CUBLAS_GEMM_DFALT
        );                        //此处的h_C是按列存储的C

#endif

        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime_cublas, start, stop);

        //将Device端计算完的结果传输会Host端  cublas方式
        cublasGetMatrixAsync(A_ROW, B_COL, sizeof(*h_C0), d_C0, A_ROW, h_C0, A_ROW, stream[0]);
        cublasGetMatrixAsync(A_ROW, B_COL, sizeof(*h_C1), d_C1, A_ROW, h_C1, A_ROW, stream[1]);

        //传统方式 流传输方式
        //cudaMemcpyAsync(h_C0, d_C0, sizeof(T2)* A_ROW* B_COL, cudaMemcpyDeviceToHost, stream[0]);
        //cudaMemcpyAsync(h_C1, d_C1, sizeof(T2)* A_ROW* B_COL, cudaMemcpyDeviceToHost, stream[1]);
    }

    for (int i = 0; i < 2; i++)
    {
        cudaStreamSynchronize(stream[i]);
    }

    //计时结束
    TIMER_STOP(_X);
    /*
    cudaDeviceSynchronize();                                                                             
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    */
    //打印结果
    cout << "cublas_kernel_async GPU多流传输、计算花费了: " << TIMER_MSEC(_X) << " ms " << "\n";

    //两个流，按理说得除以二才是一个流得平均计算时间
    std::cout << "cublas_kernel_async GPU计算花费了：" << elapsedTime_cublas * N << " ms " << std::endl<< std::endl;

#if defined(USE_FLOAT_T)
    // 按行优先顺序读取h_C相当于做了CT的结果
    //Matrixshow<T2>("计算结果C的转置的值 ( C = A*B )", A_ROW, B_COL, h_C0, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C0, h_CC, 1);
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C1, h_CC, 1);
#endif 

#elif defined(USE_FLOAT_N)
    //按行读取h_C相当于做了CTT=C的结果
    //Matrixshow<T2>("计算结果C的值 ( C^T = (B^T*A^T) = (B*A)^T )", A_ROW, B_COL, h_C, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C0, h_CC, 0);
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C1, h_CC, 0);
#endif 

#elif defined(USE_DOUBLE_T)
    // 按行优先顺序读取h_C相当于做了CT的结果
    //Matrixshow<T2>("计算结果C的转置的值 ( C = A*B )", A_ROW, B_COL, h_C0, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C0, h_CC, 1);
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C1, h_CC, 1);
#endif 

#elif defined(USE_DOUBLE_N)
    //按行读取h_C相当于做了CTT=C的结果
    //Matrixshow<T2>("计算结果C的值 ( C^T = (B^T*A^T) = (B*A)^T )", A_ROW, B_COL, h_C, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C0, h_CC, 0);
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C1, h_CC, 0);
#endif 


#elif defined(USE_INT8_T)
    // 按行优先顺序读取h_C相当于做了CT的结果
    //Matrixshow<T2>("计算结果C的转置的值 ( C = A*B )", A_ROW, B_COL, h_C, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C0, h_CC, 1);
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C1, h_CC, 1);
#endif 


#elif defined(USE_INT8_N)
    //按行读取h_C相当于做了CTT=C的结果
    //Matrixshow<T2>("计算结果C的值 ( C^T = (B^T*A^T) = (B*A)^T )", A_ROW, B_COL, h_C, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C0, h_CC, 0);
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C1, h_CC, 0);
#endif 

#endif

    //释放内存
    cudaFree(d_A0);
    cudaFree(d_B0);
    cudaFree(d_C0);
    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_C1);
    /*
    cudaFreeHost(d_A0);
    cudaFreeHost(d_B0);
    cudaFreeHost(d_C0);
    cudaFreeHost(d_A1);
    cudaFreeHost(d_B1);
    cudaFreeHost(d_C1);
    */
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C0);
    cudaFreeHost(h_C1);
    cudaFreeHost(h_CC);
    for (int i = 0; i < 2; i++)
    {
        cublasDestroy(handle[i]);
        cudaStreamDestroy(stream[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


template <typename T1, typename T2>
void cublas_kernel()
{
    // 定义状态变量
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //存储于内存中的矩阵
    T1* h_A, * h_B;
    T2* h_C, * h_CC;

    //在内存中开辟空间
    cudaHostAlloc((void**)&h_A, sizeof(T1) * A_ROW * A_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_B, sizeof(T1) * B_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_C, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_CC, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);

    MatrixINIT<T1>(A_ROW, A_COL, h_A);
    MatrixINIT<T1>(B_ROW, B_COL, h_B);

    // 打印待测试的矩阵
#if defined(USE_FLOAT_T)
    Matrixshow<T1>("A", A_ROW, A_COL, h_A, 0);
    Matrixshow<T1>("B", B_ROW, B_COL, h_B, 0);

#elif defined(USE_FLOAT_N)
    Matrixshow<T1>("A", A_ROW, A_COL, h_A, 0);
    Matrixshow<T1>("B", B_ROW, B_COL, h_B, 0);

#elif defined(USE_DOUBLE_T)
    Matrixshow<T1>("A", A_ROW, A_COL, h_A, 0);
    Matrixshow<T1>("B", B_ROW, B_COL, h_B, 0);

#elif defined(USE_DOUBLE_N)
    Matrixshow<T1>("A", A_ROW, A_COL, h_A, 0);
    Matrixshow<T1>("B", B_ROW, B_COL, h_B, 0);

#elif defined(USE_INT8_T)
    Matrixshow<T1>("A", A_ROW, A_COL, h_A, 0, 0, "char");
    Matrixshow<T1>("B", B_ROW, B_COL, h_B, 0, 0, "char");

#elif defined(USE_INT8_N)
    Matrixshow<T1>("A", A_ROW, A_COL, h_A, 0, 0, "char");
    Matrixshow<T1>("B", B_ROW, B_COL, h_B, 0, 0, "char");
#endif

    //存储于显存中的矩阵
    T1* d_A, * d_B;
    T2* d_C;

    cudaMalloc((void**)&d_A, sizeof(T1) * A_ROW * A_COL);
    cudaMalloc((void**)&d_B, sizeof(T1) * B_ROW * B_COL);
    cudaMalloc((void**)&d_C, sizeof(T2) * A_ROW * B_COL);
  
    /*
    cudaHostAlloc((void**)&d_A, sizeof(T1) * A_ROW * A_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&d_B, sizeof(T1) * B_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&d_C, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);
    */


    //创建流对象，用于任务级(Grid)同步
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cublasSetStream(handle, stream);

    const T2 a = 1.0f, b = 0.0f;

    //计时开始
    TIMER_START(_X);
    //cudaEventRecord(start, 0);

    for (int i = 0; i < N  ; i++)
    {
        //数据从Host端拷贝到Device端、 cubals方式
        /*
        cublasSetMatrix(A_ROW, A_COL, sizeof(*h_A), h_A, A_ROW, d_A, A_ROW);
        cublasSetMatrix(B_ROW, B_COL, sizeof(*h_B), h_B, B_ROW, d_B, B_ROW);
        */

        cublasSetMatrixAsync(A_ROW, A_COL, sizeof(*h_A), h_A, A_ROW, d_A, A_ROW, stream);
        cublasSetMatrixAsync(B_ROW, B_COL, sizeof(*h_B), h_B, B_ROW, d_B, B_ROW, stream);

        //数据从Host端拷贝到Device端、 传统方式
        //cudaMemcpy(d_A, H_A, sizeof(T1) * A_ROW * A_COL, cudaMemcpyHostToDevice); 
        //cudaMemcpy(d_B, H_B, sizeof(T1) * B_ROW * B_COL, cudaMemcpyHostToDevice);

        //单独计算核函数运算时间
        cudaEventRecord(start, 0);

#if defined(USE_FLOAT_N)
        cublasSgemm(
            handle,
            CUBLAS_OP_N,   //矩阵A的属性参数，不转置，按列优先
            CUBLAS_OP_N,   //矩阵B的属性参数，不转置，按列优先
            B_COL,          //矩阵B^T、C^T的行数
            A_ROW,          //矩阵A^T、C^T的列数
            B_ROW,          //B^T的列数，A^T的行数，此处也可为A_COL,一样的
            &a,             //alpha的值
            d_B,            //左矩阵，为B^T
            B_COL,          //B^T的leading dimension，按列优先，则leading dimension为B^T的行数(B的列数)
            d_A,            //右矩阵，为A^T
            A_COL,          //A^T的leading dimension，按列优先，则leading dimension为A^T的行数(A的列数)
            &b,             //beta的值
            d_C,            //结果矩阵C
            B_COL           //C^T的leading dimension，C^T矩阵一定按列优先，则leading dimension为C^T的行数(C的列数)
        );//此处的h_C是按列存储的C^T


#elif defined(USE_FLOAT_T)
        cublasSgemm(
            handle,
            CUBLAS_OP_T,   //矩阵A的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            CUBLAS_OP_T,   //矩阵B的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            A_ROW,          //矩阵A、C的行数
            B_COL,          //矩阵B、C的列数
            A_COL,          //A的列数，B的行数，此处也可为B_ROW一样的
            &a,             //alpha的值
            d_A,            //左矩阵，为A
            A_COL,          //A的leading dimension，按列优先，则leading dimension为（A^T的行数）A的列数
            d_B,            //右矩阵，为B
            B_COL,          //B的leading dimension，按列优先，则leading dimension为（B^T的行数）A的列数
            &b,             //beta的值
            d_C,            //结果矩阵C
            A_ROW           //C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
        );//此处的h_C是按列存储的C

#elif defined(USE_DOUBLE_N)
        cublasDgemm(
            handle,
            CUBLAS_OP_N,   //矩阵A的属性参数，不转置，按列优先
            CUBLAS_OP_N,   //矩阵B的属性参数，不转置，按列优先
            B_COL,          //矩阵B^T、C^T的行数
            A_ROW,          //矩阵A^T、C^T的列数
            B_ROW,          //B^T的列数，A^T的行数，此处也可为A_COL,一样的
            &a,             //alpha的值
            d_B,            //左矩阵，为B^T
            B_COL,          //B^T的leading dimension，按列优先，则leading dimension为B^T的行数(B的列数)
            d_A,            //右矩阵，为A^T
            A_COL,          //A^T的leading dimension，按列优先，则leading dimension为A^T的行数(A的列数)
            &b,             //beta的值
            d_C,            //结果矩阵C
            B_COL           //C^T的leading dimension，C^T矩阵一定按列优先，则leading dimension为C^T的行数(C的列数)
        );//此处的h_C是按列存储的C^T

#elif defined(USE_DOUBLE_T)
        cublasDgemm(
            handle,
            CUBLAS_OP_T,   //矩阵A的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            CUBLAS_OP_T,   //矩阵B的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            A_ROW,          //矩阵A、C的行数
            B_COL,          //矩阵B、C的列数
            A_COL,          //A的列数，B的行数，此处也可为B_ROW一样的
            &a,             //alpha的值
            d_A,            //左矩阵，为A
            A_COL,          //A的leading dimension，按列优先，则leading dimension为（A^T的行数）A的列数
            d_B,            //右矩阵，为B
            B_COL,          //B的leading dimension，按列优先，则leading dimension为（B^T的行数）A的列数
            &b,             //beta的值
            d_C,            //结果矩阵C
            A_ROW           //C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
        );//此处的h_C是按列存储的C
#elif defined(USE_INT8_N)
        cublasGemmEx(handle,  //句柄
            CUBLAS_OP_N,      //矩阵A的属性参数，不转置，按列优先
            CUBLAS_OP_N,      //矩阵B的属性参数，不转置，按列优先
            B_COL,            //矩阵B^T、C^T的行数
            A_ROW,            //矩阵A^T、C^T的列数
            B_ROW,            //B^T的列数，A^T的行数，此处也可为A_COL,一样的
            &a,               //alpha的值
            d_B,              //左矩阵，为B^T
            CUDA_R_8I,        //A矩阵计算模式，int8型
            B_COL,            //B^T的leading dimension，按列优先，则leading dimension为B^T的行数(B的列数)
            d_A,              //右矩阵，为A^T
            CUDA_R_8I,        //B矩阵计算模式，int8型
            A_COL,            //A^T的leading dimension，按列优先，则leading dimension为A^T的行数(A的列数)
            &b,               //乘法因子beta
            d_C,              //C结果矩阵
            CUDA_R_32I,       //C矩阵计算模式，int32型
            B_COL,             //C^T的leading dimension，C^T矩阵一定按列优先，则leading dimension为C^T的行数(C的列数)
            CUDA_R_32I,       //计算模式，int32模式
            //CUBLAS_GEMM_ALGO0    //算法参数
            CUBLAS_GEMM_DFALT
        );                    //此处的h_C是按列存储的C^T

#elif defined(USE_INT8_T)
        cublasGemmEx(handle,      //句柄
            CUBLAS_OP_T,          //矩阵A的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            CUBLAS_OP_T,          //矩阵B的属性参数，还是按列优先读取，但是在计算前，转置，变成正常c/c++的方式
            A_ROW,                //矩阵A、C的行数
            B_COL,                //矩阵B、C的列数
            A_COL,                //A的列数，B的行数，此处也可为B_ROW一样的
            &a,                   //运算式的 α 值
            d_A,                  //A矩阵
            CUDA_R_8I,            //A矩阵计算模式，int8型
            A_COL,                //A的leading dimension，按行优先存储，读取还是列优先，则leading dimension为（A^T的行数）A的列数
            d_B,                  //B矩阵
            CUDA_R_8I,            //B矩阵计算模式，int8型
            B_COL,                //B的leading dimension，按行优先存储，读取还是列优先，则leading dimension为（B^T的行数）A的列数
            &b,                   //乘法因子beta
            d_C,                  //C结果矩阵
            CUDA_R_32I,           //C矩阵计算模式，int32型
            A_ROW,                //C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
            CUDA_R_32I,           //计算模式，int32模式
            //CUBLAS_GEMM_ALGO2     //算法参数
            CUBLAS_GEMM_DFALT
        );                        //此处的h_C是按列存储的C

#endif

        //计时结束
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);

        //TIMER_STOP(_X);
        //cout << "GPU耗费了: " << TIMER_MSEC(_X) << " ms " << "\n";
        // 
        //将Device端计算完的结果传输会Host端  cublas方式
        //cublasGetMatrix(A_ROW, B_COL, sizeof(*h_C), d_C, A_ROW, h_C, A_ROW);
        cublasGetMatrixAsync(A_ROW, B_COL, sizeof(*h_C), d_C, A_ROW, h_C, A_ROW, stream);
        //传统方式
        //cudaMemcpy(H_C, d_C, sizeof(T2) * A_ROW * B_COL, cudaMemcpyDeviceToHost);
    }
    TIMER_STOP(_X);
    /*
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime_cublas, start, stop);
    */
    //打印结果
    cout << "cublas_kernel GPU传输、计算花费了:  " << TIMER_MSEC(_X) << " ms " << "\n";
    //std::cout<< "GPU传输、计算花费了：" << elapsedTime_cublas << " ms" << std::endl;
    std::cout << "cublas_kernel GPU计算花费了：" << elapsedTime * N<< " ms" << std::endl<< std::endl;


#if defined(USE_FLOAT_T)
    // 按行优先顺序读取h_C相当于做了CT的结果
    Matrixshow<T2>("计算结果C的转置的值 ( C = A*B )", A_ROW, B_COL, h_C, 0, 1);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C, h_CC, 1);
#endif 
    

#elif defined(USE_FLOAT_N)
    //按行读取h_C相当于做了CTT=C的结果
    Matrixshow<T2>("计算结果C的值 ( C^T = (B^T*A^T) = (B*A)^T )", A_ROW, B_COL, h_C, 0, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C, h_CC, 0);
#endif 

#elif defined(USE_DOUBLE_T)
        // 按行优先顺序读取h_C相当于做了CT的结果
        Matrixshow<T2>("计算结果C的转置的值 ( C = A*B )", A_ROW, B_COL, h_C, 0, 1);
        cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C, h_CC, 1);
#endif 

#elif defined(USE_DOUBLE_N)
    //按行读取h_C相当于做了CTT=C的结果
    Matrixshow<T2>("计算结果C的值 ( C^T = (B^T*A^T) = (B*A)^T )", A_ROW, B_COL, h_C, 0, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C, h_CC, 0);
#endif 

#elif defined(USE_INT8_T)
    // 按行优先顺序读取h_C相当于做了CT的结果
    Matrixshow<T2>("计算结果C的转置的值 ( C = A*B )", A_ROW, B_COL, h_C, 0, 1);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C, h_CC, 1);
#endif 

#elif defined(USE_INT8_N)
    //按行读取h_C相当于做了CTT=C的结果
    //Matrixshow<T2>("计算结果C的值 ( C^T = (B^T*A^T) = (B*A)^T )", A_ROW, B_COL, h_C, 0, 0);
    cout << endl;
#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C, h_CC, 0);
#endif 

#endif

    //释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    /*
    cudaFreeHost(d_A);
    cudaFreeHost(d_B);
    cudaFreeHost(d_C);
    */
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_CC);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}

int main()
{
    warm_up <<<1, 4 >>> ();
    
    cudaDeviceSynchronize();
    std::cout << "\n";

#if defined(USE_MY_INT)
    My_kernel<int, int>();
    //My_kernel_2D<int, int>();

#elif defined(USE_MY_FLOAT)
    My_kernel<float, float>();
    //My_kernel_2D<float, float>();

#elif defined(USE_MY_DOUBLE)
    My_kernel<double,double>();
    //My_kernel_2D<double, double>();
#endif

#if defined(USE_ISPC_INT)
    cpu_ispc<int, int>();
#elif defined(USE_ISPC_FLOAT)
    cpu_ispc<float, float>();
#elif defined(USE_ISPC_DOUBLE)
    cpu_ispc<double, double>();
#endif


#if defined(USE_INT8_T)
    cublas_kernel<char, int>();
    cublas_kernel_asny<char, int>();
#elif defined(USE_INT8_N)
    cublas_kernel<char, int>();
    cublas_kernel_asny<char, int>();

#elif defined(USE_FLOAT_T)
    cublas_kernel<float, float>();
    cublas_kernel_asny<float, float>();
#elif defined(USE_FLOAT_N)
    cublas_kernel<float, float>();
    cublas_kernel_asny<float, float>();

#elif defined(USE_DOUBLE_T)
    cublas_kernel<double, double>();
    cublas_kernel_asny<double, double>();

#elif defined(USE_DOUBLE_N)
    cublas_kernel<double, double>();
    cublas_kernel_asny<double, double>();
#endif
    return 0;
}