
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
    //��ǰ�̶߳�Ӧ�ľ���C��Ԫ��λ��
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;
    //int I = (COL_A + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int I = (COL_A + BLOCK_SIZE - 1) / BLOCK_SIZE;

    T2 t=0.0f, Csub = 0.0f, comp = 0.0f;
 
    __shared__ float As[BLOCK_SIZE+1][BLOCK_SIZE+1];
    __shared__ float Bs[BLOCK_SIZE+1][BLOCK_SIZE+1];

    //ÿ��Block��������A��һ���п��B��һ���п�
    //ÿ���߳���Ҫ����һ�к�һ�е��ڻ������⻹����Ϊ��ǰѭ����������Ŀ����һ��Ԫ�ص������ڴ���
    //��������ȡ��
    for (int i = 0; i < I; i++) {
        
        if (row < ROW_A && i * BLOCK_SIZE + tx < COL_A)
            As[ty][tx] = A[row * COL_A + i * BLOCK_SIZE + tx];//���м��㵥Ԫͬʱ���أ����������forѭ����As��Bs�����������
        else
            As[ty][tx] = 0;

        if (col < COL_B && i * BLOCK_SIZE + ty < ROW_B)
            Bs[ty][tx] = B[(i * BLOCK_SIZE + ty) * COL_B + col];
        else
            Bs[ty][tx] = 0;

        //��ͬһ���еĲ�ͬ�߳�ָ����ͬ������֤�����ڴ��о�����Ԫ��ȫ������
        __syncthreads();//���߳�ִ�е��˺���ʱ�ȴ���ֱ��ȫ���߳�ͬ��


        //Kahan's Summation Formula
        //��Ȼ���ѭ��������Block�ģ��������ڲ�ѭ��ֻ������������ĳ�к�ĳ�е�
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

    //��ǰ�̶߳�Ӧ�ľ���C��Ԫ��λ��
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

    //����CPU�ϵĴ洢�ռ�
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

    //����GPU�ϵĴ洢�ռ�
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

    //������������������(Grid)ͬ��
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    //��ʱ��ʼ
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
    cout <<"mykernel GPU���䡢���㻨����: " << TIMER_MSEC(_X) << " ms " << "\n";

    std::cout <<"mykernel GPU���㻨���ˣ�"<<elapsedTime_mykernel * N<< " ms" << std::endl;
    //Matrixshow<T2>("����������C��ֵ", A_ROW, B_COL, h_c, 0);
    cout << endl;

    //�������Ƿ���ȷ
#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_a, h_b, A_ROW, A_COL, B_COL, h_c, h_cc, 0);
#endif 

    //�����ڴ�
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
//��������Ƿ�������
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

    //������������������(Grid)ͬ��
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
     
    unsigned int grid_rows = (A_ROW + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (B_COL + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(grid_rows, grid_cols);
    dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);

    //��ʱ��ʼ
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
  
    //��ʱ����
    TIMER_STOP(_X);
    cout << "my_kernel_2D GPU���䡢���㻨����: " << TIMER_MSEC(_X) << " ms " << "\n";
    std::cout<< "my_kernel_2D GPU�����ˣ�" << elapsedTime_mykernel * N << " ms " << std::endl;

    //Matrixshow<T2>("����������C��ֵ", A_ROW, B_COL, h_c, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_a, h_b, A_ROW, A_COL, B_COL, h_c, h_cc, 0);
#endif 

    //�����ڴ�
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
    // ����״̬����
    cublasHandle_t handle[2];
    for (int i = 0; i < 2; i++)
    {
        cublasCreate(&handle[i]);
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //�洢���ڴ��еľ���
    T1* h_A, * h_B;
    T2* h_C0, * h_C1, * h_CC;

    //���ڴ��п��ٿռ�
    cudaHostAlloc((void**)&h_A, sizeof(T1) * A_ROW * A_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_B, sizeof(T1) * B_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_C0, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_C1, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_CC, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);



    MatrixINIT<T1>(A_ROW, A_COL, h_A);
    MatrixINIT<T1>(B_ROW, B_COL, h_B);

    // ��ӡ�����Եľ���
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

    //�洢���Դ��еľ���
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

    //������������
    cudaStream_t stream[2];
    for (int i = 0; i < 2; i++)
    {
        cudaStreamCreate(&stream[i]);
    }

    //��cublas�������
    for (int i = 0; i < 2; i++)
    {
        cublasSetStream(handle[i], stream[i]);
    }

    const T2 a = 1.0f, b = 0.0f;

    //���ݴ�Host�˿�����Device�ˡ� ��ͳ��ʽ
    //cudaMemcpy(d_A, H_A, sizeof(T1) * A_ROW * A_COL, cudaMemcpyHostToDevice); 
    //cudaMemcpy(d_B, H_B, sizeof(T1) * B_ROW * B_COL, cudaMemcpyHostToDevice);

     //��ʱ��ʼ
    TIMER_START(_X);
    //cudaEventRecord(start, 0);

    for (int i = 0; i < N / 2; i++)
    {
        /*
        //���ݴ�Host�˿�����Device�ˡ� �������䷽ʽ
        cudaMemcpyAsync(d_A0, h_A, sizeof(T1) * A_ROW * A_COL, cudaMemcpyHostToDevice, stream[0]);
        cudaMemcpyAsync(d_B0, h_B, sizeof(T1) * B_ROW * B_COL, cudaMemcpyHostToDevice, stream[0]);
        cudaMemcpyAsync(d_A1, h_A, sizeof(T1) * A_ROW * A_COL, cudaMemcpyHostToDevice, stream[1]);
        cudaMemcpyAsync(d_B1, h_B, sizeof(T1) * B_ROW * B_COL, cudaMemcpyHostToDevice, stream[1]);
        */

        
       //���ݴ�Host�˿�����Device�ˡ� cubals��ʽ
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
            CUBLAS_OP_N,   //����A�����Բ�������ת�ã���������
            CUBLAS_OP_N,   //����B�����Բ�������ת�ã���������
            B_COL,          //����B^T��C^T������
            A_ROW,          //����A^T��C^T������
            B_ROW,          //B^T��������A^T���������˴�Ҳ��ΪA_COL,һ����
            &a,             //alpha��ֵ
            d_B0,            //�����ΪB^T
            B_COL,          //B^T��leading dimension���������ȣ���leading dimensionΪB^T������(B������)
            d_A0,            //�Ҿ���ΪA^T
            A_COL,          //A^T��leading dimension���������ȣ���leading dimensionΪA^T������(A������)
            &b,             //beta��ֵ
            d_C0,            //�������C
            B_COL           //C^T��leading dimension��C^T����һ���������ȣ���leading dimensionΪC^T������(C������)
        );//�˴���h_C�ǰ��д洢��C^T

        cublasSgemm(
            handle[1],
            CUBLAS_OP_N,   //����A�����Բ�������ת�ã���������
            CUBLAS_OP_N,   //����B�����Բ�������ת�ã���������
            B_COL,          //����B^T��C^T������
            A_ROW,          //����A^T��C^T������
            B_ROW,          //B^T��������A^T���������˴�Ҳ��ΪA_COL,һ����
            &a,             //alpha��ֵ
            d_B1,            //�����ΪB^T
            B_COL,          //B^T��leading dimension���������ȣ���leading dimensionΪB^T������(B������)
            d_A1,            //�Ҿ���ΪA^T
            A_COL,          //A^T��leading dimension���������ȣ���leading dimensionΪA^T������(A������)
            &b,             //beta��ֵ
            d_C1,            //�������C
            B_COL           //C^T��leading dimension��C^T����һ���������ȣ���leading dimensionΪC^T������(C������)
        );//�˴���h_C�ǰ��д洢��C^T

#elif defined(USE_FLOAT_T)
        cublasSgemm(
            handle[0],
            CUBLAS_OP_T,   //����A�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            CUBLAS_OP_T,   //����B�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            A_ROW,          //����A��C������
            B_COL,          //����B��C������
            A_COL,          //A��������B���������˴�Ҳ��ΪB_ROWһ����
            &a,             //alpha��ֵ
            d_A0,            //�����ΪA
            A_COL,          //A��leading dimension���������ȣ���leading dimensionΪ��A^T��������A������
            d_B0,            //�Ҿ���ΪB
            B_COL,          //B��leading dimension���������ȣ���leading dimensionΪ��B^T��������A������
            &b,             //beta��ֵ
            d_C0,            //�������C
            A_ROW           //C��leading dimension��C����һ���������ȣ���leading dimensionΪC������
        );//�˴���h_C�ǰ��д洢��C

        cublasSgemm(
            handle[1],
            CUBLAS_OP_T,   //����A�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            CUBLAS_OP_T,   //����B�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            A_ROW,          //����A��C������
            B_COL,          //����B��C������
            A_COL,          //A��������B���������˴�Ҳ��ΪB_ROWһ����
            &a,             //alpha��ֵ
            d_A1,            //�����ΪA
            A_COL,          //A��leading dimension���������ȣ���leading dimensionΪ��A^T��������A������
            d_B1,            //�Ҿ���ΪB
            B_COL,          //B��leading dimension���������ȣ���leading dimensionΪ��B^T��������A������
            &b,             //beta��ֵ
            d_C1,            //�������C
            A_ROW           //C��leading dimension��C����һ���������ȣ���leading dimensionΪC������
        );//�˴���h_C�ǰ��д洢��C

#elif defined(USE_DOUBLE_T)
        cublasDgemm(
            handle[0],
            CUBLAS_OP_T,   //����A�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            CUBLAS_OP_T,   //����B�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            A_ROW,          //����A��C������
            B_COL,          //����B��C������
            A_COL,          //A��������B���������˴�Ҳ��ΪB_ROWһ����
            &a,             //alpha��ֵ
            d_A0,            //�����ΪA
            A_COL,          //A��leading dimension���������ȣ���leading dimensionΪ��A^T��������A������
            d_B0,            //�Ҿ���ΪB
            B_COL,          //B��leading dimension���������ȣ���leading dimensionΪ��B^T��������A������
            &b,             //beta��ֵ
            d_C0,            //�������C
            A_ROW           //C��leading dimension��C����һ���������ȣ���leading dimensionΪC������
        );//�˴���h_C�ǰ��д洢��C

        cublasDgemm(
            handle[1],
            CUBLAS_OP_T,   //����A�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            CUBLAS_OP_T,   //����B�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            A_ROW,          //����A��C������
            B_COL,          //����B��C������
            A_COL,          //A��������B���������˴�Ҳ��ΪB_ROWһ����
            &a,             //alpha��ֵ
            d_A1,            //�����ΪA
            A_COL,          //A��leading dimension���������ȣ���leading dimensionΪ��A^T��������A������
            d_B1,            //�Ҿ���ΪB
            B_COL,          //B��leading dimension���������ȣ���leading dimensionΪ��B^T��������A������
            &b,             //beta��ֵ
            d_C1,            //�������C
            A_ROW           //C��leading dimension��C����һ���������ȣ���leading dimensionΪC������
        );//�˴���h_C�ǰ��д洢��C

#elif defined(USE_DOUBLE_N)
        cublasDgemm(
            handle[0],
            CUBLAS_OP_N,   //����A�����Բ�������ת�ã���������
            CUBLAS_OP_N,   //����B�����Բ�������ת�ã���������
            B_COL,          //����B^T��C^T������
            A_ROW,          //����A^T��C^T������
            B_ROW,          //B^T��������A^T���������˴�Ҳ��ΪA_COL,һ����
            &a,             //alpha��ֵ
            d_B0,            //�����ΪB^T
            B_COL,          //B^T��leading dimension���������ȣ���leading dimensionΪB^T������(B������)
            d_A0,            //�Ҿ���ΪA^T
            A_COL,          //A^T��leading dimension���������ȣ���leading dimensionΪA^T������(A������)
            &b,             //beta��ֵ
            d_C0,            //�������C
            B_COL           //C^T��leading dimension��C^T����һ���������ȣ���leading dimensionΪC^T������(C������)
        );//�˴���h_C�ǰ��д洢��C^T

        cublasDgemm(
            handle[1],
            CUBLAS_OP_N,   //����A�����Բ�������ת�ã���������
            CUBLAS_OP_N,   //����B�����Բ�������ת�ã���������
            B_COL,          //����B^T��C^T������
            A_ROW,          //����A^T��C^T������
            B_ROW,          //B^T��������A^T���������˴�Ҳ��ΪA_COL,һ����
            &a,             //alpha��ֵ
            d_B1,            //�����ΪB^T
            B_COL,          //B^T��leading dimension���������ȣ���leading dimensionΪB^T������(B������)
            d_A1,            //�Ҿ���ΪA^T
            A_COL,          //A^T��leading dimension���������ȣ���leading dimensionΪA^T������(A������)
            &b,             //beta��ֵ
            d_C1,            //�������C
            B_COL           //C^T��leading dimension��C^T����һ���������ȣ���leading dimensionΪC^T������(C������)
        );//�˴���h_C�ǰ��д洢��C^T


#elif defined(USE_INT8_N)
        cublasGemmEx(handle[0],  //���
            CUBLAS_OP_N,      //����A�����Բ�������ת�ã���������
            CUBLAS_OP_N,      //����B�����Բ�������ת�ã���������
            B_COL,            //����B^T��C^T������
            A_ROW,            //����A^T��C^T������
            B_ROW,            //B^T��������A^T���������˴�Ҳ��ΪA_COL,һ����
            &a,               //alpha��ֵ
            d_B0,              //�����ΪB^T
            CUDA_R_8I,        //A�������ģʽ��int8��
            B_COL,            //B^T��leading dimension���������ȣ���leading dimensionΪB^T������(B������)
            d_A0,              //�Ҿ���ΪA^T
            CUDA_R_8I,        //B�������ģʽ��int8��
            A_COL,            //A^T��leading dimension���������ȣ���leading dimensionΪA^T������(A������)
            &b,               //�˷�����beta
            d_C0,              //C�������
            CUDA_R_32I,       //C�������ģʽ��int32��
            B_COL,             //C^T��leading dimension��C^T����һ���������ȣ���leading dimensionΪC^T������(C������)
            CUDA_R_32I,       //����ģʽ��int32ģʽ
            //CUBLAS_GEMM_ALGO0    //�㷨����
            CUBLAS_GEMM_DFALT
        );                    //�˴���h_C�ǰ��д洢��C^T

        cublasGemmEx(handle[1],  //���
            CUBLAS_OP_N,      //����A�����Բ�������ת�ã���������
            CUBLAS_OP_N,      //����B�����Բ�������ת�ã���������
            B_COL,            //����B^T��C^T������
            A_ROW,            //����A^T��C^T������
            B_ROW,            //B^T��������A^T���������˴�Ҳ��ΪA_COL,һ����
            &a,               //alpha��ֵ
            d_B1,              //�����ΪB^T
            CUDA_R_8I,        //A�������ģʽ��int8��
            B_COL,            //B^T��leading dimension���������ȣ���leading dimensionΪB^T������(B������)
            d_A1,              //�Ҿ���ΪA^T
            CUDA_R_8I,        //B�������ģʽ��int8��
            A_COL,            //A^T��leading dimension���������ȣ���leading dimensionΪA^T������(A������)
            &b,               //�˷�����beta
            d_C1,              //C�������
            CUDA_R_32I,       //C�������ģʽ��int32��
            B_COL,             //C^T��leading dimension��C^T����һ���������ȣ���leading dimensionΪC^T������(C������)
            CUDA_R_32I,       //����ģʽ��int32ģʽ
            //CUBLAS_GEMM_ALGO0    //�㷨����
            CUBLAS_GEMM_DFALT
        );                    //�˴���h_C�ǰ��д洢��C^T

#elif defined(USE_INT8_T)
        cublasGemmEx(handle[0],      //���
            CUBLAS_OP_T,          //����A�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            CUBLAS_OP_T,          //����B�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            A_ROW,                //����A��C������
            B_COL,                //����B��C������
            A_COL,                //A��������B���������˴�Ҳ��ΪB_ROWһ����
            &a,                   //����ʽ�� �� ֵ
            d_A0,                  //A����
            CUDA_R_8I,            //A�������ģʽ��int8��
            A_COL,                //A��leading dimension���������ȴ洢����ȡ���������ȣ���leading dimensionΪ��A^T��������A������
            d_B0,                  //B����
            CUDA_R_8I,            //B�������ģʽ��int8��
            B_COL,                //B��leading dimension���������ȴ洢����ȡ���������ȣ���leading dimensionΪ��B^T��������A������
            &b,                   //�˷�����beta
            d_C0,                  //C�������
            CUDA_R_32I,           //C�������ģʽ��int32��
            A_ROW,                //C��leading dimension��C����һ���������ȣ���leading dimensionΪC������
            CUDA_R_32I,           //����ģʽ��int32ģʽ
            //CUBLAS_GEMM_ALGO2     //�㷨����
            CUBLAS_GEMM_DFALT
        );                        //�˴���h_C�ǰ��д洢��C


        cublasGemmEx(handle[1],      //���
            CUBLAS_OP_T,          //����A�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            CUBLAS_OP_T,          //����B�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            A_ROW,                //����A��C������
            B_COL,                //����B��C������
            A_COL,                //A��������B���������˴�Ҳ��ΪB_ROWһ����
            &a,                   //����ʽ�� �� ֵ
            d_A1,                  //A����
            CUDA_R_8I,            //A�������ģʽ��int8��
            A_COL,                //A��leading dimension���������ȴ洢����ȡ���������ȣ���leading dimensionΪ��A^T��������A������
            d_B1,                  //B����
            CUDA_R_8I,            //B�������ģʽ��int8��
            B_COL,                //B��leading dimension���������ȴ洢����ȡ���������ȣ���leading dimensionΪ��B^T��������A������
            &b,                   //�˷�����beta
            d_C1,                  //C�������
            CUDA_R_32I,           //C�������ģʽ��int32��
            A_ROW,                //C��leading dimension��C����һ���������ȣ���leading dimensionΪC������
            CUDA_R_32I,           //����ģʽ��int32ģʽ
            //CUBLAS_GEMM_ALGO2     //�㷨����
            CUBLAS_GEMM_DFALT
        );                        //�˴���h_C�ǰ��д洢��C

#endif

        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime_cublas, start, stop);

        //��Device�˼�����Ľ�������Host��  cublas��ʽ
        cublasGetMatrixAsync(A_ROW, B_COL, sizeof(*h_C0), d_C0, A_ROW, h_C0, A_ROW, stream[0]);
        cublasGetMatrixAsync(A_ROW, B_COL, sizeof(*h_C1), d_C1, A_ROW, h_C1, A_ROW, stream[1]);

        //��ͳ��ʽ �����䷽ʽ
        //cudaMemcpyAsync(h_C0, d_C0, sizeof(T2)* A_ROW* B_COL, cudaMemcpyDeviceToHost, stream[0]);
        //cudaMemcpyAsync(h_C1, d_C1, sizeof(T2)* A_ROW* B_COL, cudaMemcpyDeviceToHost, stream[1]);
    }

    for (int i = 0; i < 2; i++)
    {
        cudaStreamSynchronize(stream[i]);
    }

    //��ʱ����
    TIMER_STOP(_X);
    /*
    cudaDeviceSynchronize();                                                                             
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    */
    //��ӡ���
    cout << "cublas_kernel_async GPU�������䡢���㻨����: " << TIMER_MSEC(_X) << " ms " << "\n";

    //������������˵�ó��Զ�����һ������ƽ������ʱ��
    std::cout << "cublas_kernel_async GPU���㻨���ˣ�" << elapsedTime_cublas * N << " ms " << std::endl<< std::endl;

#if defined(USE_FLOAT_T)
    // ��������˳���ȡh_C�൱������CT�Ľ��
    //Matrixshow<T2>("������C��ת�õ�ֵ ( C = A*B )", A_ROW, B_COL, h_C0, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C0, h_CC, 1);
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C1, h_CC, 1);
#endif 

#elif defined(USE_FLOAT_N)
    //���ж�ȡh_C�൱������CTT=C�Ľ��
    //Matrixshow<T2>("������C��ֵ ( C^T = (B^T*A^T) = (B*A)^T )", A_ROW, B_COL, h_C, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C0, h_CC, 0);
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C1, h_CC, 0);
#endif 

#elif defined(USE_DOUBLE_T)
    // ��������˳���ȡh_C�൱������CT�Ľ��
    //Matrixshow<T2>("������C��ת�õ�ֵ ( C = A*B )", A_ROW, B_COL, h_C0, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C0, h_CC, 1);
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C1, h_CC, 1);
#endif 

#elif defined(USE_DOUBLE_N)
    //���ж�ȡh_C�൱������CTT=C�Ľ��
    //Matrixshow<T2>("������C��ֵ ( C^T = (B^T*A^T) = (B*A)^T )", A_ROW, B_COL, h_C, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C0, h_CC, 0);
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C1, h_CC, 0);
#endif 


#elif defined(USE_INT8_T)
    // ��������˳���ȡh_C�൱������CT�Ľ��
    //Matrixshow<T2>("������C��ת�õ�ֵ ( C = A*B )", A_ROW, B_COL, h_C, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C0, h_CC, 1);
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C1, h_CC, 1);
#endif 


#elif defined(USE_INT8_N)
    //���ж�ȡh_C�൱������CTT=C�Ľ��
    //Matrixshow<T2>("������C��ֵ ( C^T = (B^T*A^T) = (B*A)^T )", A_ROW, B_COL, h_C, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C0, h_CC, 0);
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C1, h_CC, 0);
#endif 

#endif

    //�ͷ��ڴ�
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
    // ����״̬����
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //�洢���ڴ��еľ���
    T1* h_A, * h_B;
    T2* h_C, * h_CC;

    //���ڴ��п��ٿռ�
    cudaHostAlloc((void**)&h_A, sizeof(T1) * A_ROW * A_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_B, sizeof(T1) * B_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_C, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_CC, sizeof(T2) * A_ROW * B_COL, cudaHostAllocDefault);

    MatrixINIT<T1>(A_ROW, A_COL, h_A);
    MatrixINIT<T1>(B_ROW, B_COL, h_B);

    // ��ӡ�����Եľ���
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

    //�洢���Դ��еľ���
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


    //������������������(Grid)ͬ��
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cublasSetStream(handle, stream);

    const T2 a = 1.0f, b = 0.0f;

    //��ʱ��ʼ
    TIMER_START(_X);
    //cudaEventRecord(start, 0);

    for (int i = 0; i < N  ; i++)
    {
        //���ݴ�Host�˿�����Device�ˡ� cubals��ʽ
        /*
        cublasSetMatrix(A_ROW, A_COL, sizeof(*h_A), h_A, A_ROW, d_A, A_ROW);
        cublasSetMatrix(B_ROW, B_COL, sizeof(*h_B), h_B, B_ROW, d_B, B_ROW);
        */

        cublasSetMatrixAsync(A_ROW, A_COL, sizeof(*h_A), h_A, A_ROW, d_A, A_ROW, stream);
        cublasSetMatrixAsync(B_ROW, B_COL, sizeof(*h_B), h_B, B_ROW, d_B, B_ROW, stream);

        //���ݴ�Host�˿�����Device�ˡ� ��ͳ��ʽ
        //cudaMemcpy(d_A, H_A, sizeof(T1) * A_ROW * A_COL, cudaMemcpyHostToDevice); 
        //cudaMemcpy(d_B, H_B, sizeof(T1) * B_ROW * B_COL, cudaMemcpyHostToDevice);

        //��������˺�������ʱ��
        cudaEventRecord(start, 0);

#if defined(USE_FLOAT_N)
        cublasSgemm(
            handle,
            CUBLAS_OP_N,   //����A�����Բ�������ת�ã���������
            CUBLAS_OP_N,   //����B�����Բ�������ת�ã���������
            B_COL,          //����B^T��C^T������
            A_ROW,          //����A^T��C^T������
            B_ROW,          //B^T��������A^T���������˴�Ҳ��ΪA_COL,һ����
            &a,             //alpha��ֵ
            d_B,            //�����ΪB^T
            B_COL,          //B^T��leading dimension���������ȣ���leading dimensionΪB^T������(B������)
            d_A,            //�Ҿ���ΪA^T
            A_COL,          //A^T��leading dimension���������ȣ���leading dimensionΪA^T������(A������)
            &b,             //beta��ֵ
            d_C,            //�������C
            B_COL           //C^T��leading dimension��C^T����һ���������ȣ���leading dimensionΪC^T������(C������)
        );//�˴���h_C�ǰ��д洢��C^T


#elif defined(USE_FLOAT_T)
        cublasSgemm(
            handle,
            CUBLAS_OP_T,   //����A�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            CUBLAS_OP_T,   //����B�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            A_ROW,          //����A��C������
            B_COL,          //����B��C������
            A_COL,          //A��������B���������˴�Ҳ��ΪB_ROWһ����
            &a,             //alpha��ֵ
            d_A,            //�����ΪA
            A_COL,          //A��leading dimension���������ȣ���leading dimensionΪ��A^T��������A������
            d_B,            //�Ҿ���ΪB
            B_COL,          //B��leading dimension���������ȣ���leading dimensionΪ��B^T��������A������
            &b,             //beta��ֵ
            d_C,            //�������C
            A_ROW           //C��leading dimension��C����һ���������ȣ���leading dimensionΪC������
        );//�˴���h_C�ǰ��д洢��C

#elif defined(USE_DOUBLE_N)
        cublasDgemm(
            handle,
            CUBLAS_OP_N,   //����A�����Բ�������ת�ã���������
            CUBLAS_OP_N,   //����B�����Բ�������ת�ã���������
            B_COL,          //����B^T��C^T������
            A_ROW,          //����A^T��C^T������
            B_ROW,          //B^T��������A^T���������˴�Ҳ��ΪA_COL,һ����
            &a,             //alpha��ֵ
            d_B,            //�����ΪB^T
            B_COL,          //B^T��leading dimension���������ȣ���leading dimensionΪB^T������(B������)
            d_A,            //�Ҿ���ΪA^T
            A_COL,          //A^T��leading dimension���������ȣ���leading dimensionΪA^T������(A������)
            &b,             //beta��ֵ
            d_C,            //�������C
            B_COL           //C^T��leading dimension��C^T����һ���������ȣ���leading dimensionΪC^T������(C������)
        );//�˴���h_C�ǰ��д洢��C^T

#elif defined(USE_DOUBLE_T)
        cublasDgemm(
            handle,
            CUBLAS_OP_T,   //����A�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            CUBLAS_OP_T,   //����B�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            A_ROW,          //����A��C������
            B_COL,          //����B��C������
            A_COL,          //A��������B���������˴�Ҳ��ΪB_ROWһ����
            &a,             //alpha��ֵ
            d_A,            //�����ΪA
            A_COL,          //A��leading dimension���������ȣ���leading dimensionΪ��A^T��������A������
            d_B,            //�Ҿ���ΪB
            B_COL,          //B��leading dimension���������ȣ���leading dimensionΪ��B^T��������A������
            &b,             //beta��ֵ
            d_C,            //�������C
            A_ROW           //C��leading dimension��C����һ���������ȣ���leading dimensionΪC������
        );//�˴���h_C�ǰ��д洢��C
#elif defined(USE_INT8_N)
        cublasGemmEx(handle,  //���
            CUBLAS_OP_N,      //����A�����Բ�������ת�ã���������
            CUBLAS_OP_N,      //����B�����Բ�������ת�ã���������
            B_COL,            //����B^T��C^T������
            A_ROW,            //����A^T��C^T������
            B_ROW,            //B^T��������A^T���������˴�Ҳ��ΪA_COL,һ����
            &a,               //alpha��ֵ
            d_B,              //�����ΪB^T
            CUDA_R_8I,        //A�������ģʽ��int8��
            B_COL,            //B^T��leading dimension���������ȣ���leading dimensionΪB^T������(B������)
            d_A,              //�Ҿ���ΪA^T
            CUDA_R_8I,        //B�������ģʽ��int8��
            A_COL,            //A^T��leading dimension���������ȣ���leading dimensionΪA^T������(A������)
            &b,               //�˷�����beta
            d_C,              //C�������
            CUDA_R_32I,       //C�������ģʽ��int32��
            B_COL,             //C^T��leading dimension��C^T����һ���������ȣ���leading dimensionΪC^T������(C������)
            CUDA_R_32I,       //����ģʽ��int32ģʽ
            //CUBLAS_GEMM_ALGO0    //�㷨����
            CUBLAS_GEMM_DFALT
        );                    //�˴���h_C�ǰ��д洢��C^T

#elif defined(USE_INT8_T)
        cublasGemmEx(handle,      //���
            CUBLAS_OP_T,          //����A�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            CUBLAS_OP_T,          //����B�����Բ��������ǰ������ȶ�ȡ�������ڼ���ǰ��ת�ã��������c/c++�ķ�ʽ
            A_ROW,                //����A��C������
            B_COL,                //����B��C������
            A_COL,                //A��������B���������˴�Ҳ��ΪB_ROWһ����
            &a,                   //����ʽ�� �� ֵ
            d_A,                  //A����
            CUDA_R_8I,            //A�������ģʽ��int8��
            A_COL,                //A��leading dimension���������ȴ洢����ȡ���������ȣ���leading dimensionΪ��A^T��������A������
            d_B,                  //B����
            CUDA_R_8I,            //B�������ģʽ��int8��
            B_COL,                //B��leading dimension���������ȴ洢����ȡ���������ȣ���leading dimensionΪ��B^T��������A������
            &b,                   //�˷�����beta
            d_C,                  //C�������
            CUDA_R_32I,           //C�������ģʽ��int32��
            A_ROW,                //C��leading dimension��C����һ���������ȣ���leading dimensionΪC������
            CUDA_R_32I,           //����ģʽ��int32ģʽ
            //CUBLAS_GEMM_ALGO2     //�㷨����
            CUBLAS_GEMM_DFALT
        );                        //�˴���h_C�ǰ��д洢��C

#endif

        //��ʱ����
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);

        //TIMER_STOP(_X);
        //cout << "GPU�ķ���: " << TIMER_MSEC(_X) << " ms " << "\n";
        // 
        //��Device�˼�����Ľ�������Host��  cublas��ʽ
        //cublasGetMatrix(A_ROW, B_COL, sizeof(*h_C), d_C, A_ROW, h_C, A_ROW);
        cublasGetMatrixAsync(A_ROW, B_COL, sizeof(*h_C), d_C, A_ROW, h_C, A_ROW, stream);
        //��ͳ��ʽ
        //cudaMemcpy(H_C, d_C, sizeof(T2) * A_ROW * B_COL, cudaMemcpyDeviceToHost);
    }
    TIMER_STOP(_X);
    /*
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime_cublas, start, stop);
    */
    //��ӡ���
    cout << "cublas_kernel GPU���䡢���㻨����:  " << TIMER_MSEC(_X) << " ms " << "\n";
    //std::cout<< "GPU���䡢���㻨���ˣ�" << elapsedTime_cublas << " ms" << std::endl;
    std::cout << "cublas_kernel GPU���㻨���ˣ�" << elapsedTime * N<< " ms" << std::endl<< std::endl;


#if defined(USE_FLOAT_T)
    // ��������˳���ȡh_C�൱������CT�Ľ��
    Matrixshow<T2>("������C��ת�õ�ֵ ( C = A*B )", A_ROW, B_COL, h_C, 0, 1);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C, h_CC, 1);
#endif 
    

#elif defined(USE_FLOAT_N)
    //���ж�ȡh_C�൱������CTT=C�Ľ��
    Matrixshow<T2>("������C��ֵ ( C^T = (B^T*A^T) = (B*A)^T )", A_ROW, B_COL, h_C, 0, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C, h_CC, 0);
#endif 

#elif defined(USE_DOUBLE_T)
        // ��������˳���ȡh_C�൱������CT�Ľ��
        Matrixshow<T2>("������C��ת�õ�ֵ ( C = A*B )", A_ROW, B_COL, h_C, 0, 1);
        cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C, h_CC, 1);
#endif 

#elif defined(USE_DOUBLE_N)
    //���ж�ȡh_C�൱������CTT=C�Ľ��
    Matrixshow<T2>("������C��ֵ ( C^T = (B^T*A^T) = (B*A)^T )", A_ROW, B_COL, h_C, 0, 0);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C, h_CC, 0);
#endif 

#elif defined(USE_INT8_T)
    // ��������˳���ȡh_C�൱������CT�Ľ��
    Matrixshow<T2>("������C��ת�õ�ֵ ( C = A*B )", A_ROW, B_COL, h_C, 0, 1);
    cout << endl;

#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C, h_CC, 1);
#endif 

#elif defined(USE_INT8_N)
    //���ж�ȡh_C�൱������CTT=C�Ľ��
    //Matrixshow<T2>("������C��ֵ ( C^T = (B^T*A^T) = (B*A)^T )", A_ROW, B_COL, h_C, 0, 0);
    cout << endl;
#if defined(USE_CPU_COST)
    cpu_matrix_mult<T1, T2>(h_A, h_B, A_ROW, A_COL, B_COL, h_C, h_CC, 0);
#endif 

#endif

    //�ͷ��ڴ�
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