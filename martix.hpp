#pragma once

#include <device_launch_parameters.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>
#include <memory>

using namespace std;

//测试次数
#define N 500

//自己的cuda核函数
#define USE_MY_INT
//#define USE_MY_FLOAT
//#define USE_MY_DOUBLE

//cublas的矩阵计算
#define USE_INT8_T
//#define USE_INT8_N
//#define USE_FLOAT_T
//#define USE_FLOAT_N
//#define USE_DOUBLE_T
//#define USE_DOUBLE_N

//ISPC的矩阵计算
#define USE_ISPC_INT
//#define USE_ISPC_FLOAT
//#define USE_ISPC_DOUBLE

//是否开始验证矩阵正确和cpu的计算时间
#define USE_CPU_COST


//矩阵A、B、C的行数列数
int const A_ROW = 512;
int const A_COL = 512;

int const B_ROW = 512;
int const B_COL = 512;


// 用TIMER_START 定义一个变量记录开始的时间
#define TIMER_START(_X) auto _X##_start = std::chrono::system_clock::now(), _X##_stop = _X##_start
// 用TIMER_STOP 定义一个变量记录结束的时间
#define TIMER_STOP(_X) _X##_stop = std::chrono::system_clock::now()
// TIMER_MSEC 定义start到stop经历多少毫秒
#define TIMER_MSEC(_X) (1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(_X##_stop - _X##_start).count())

template <typename T>
//默认打印初始化矩阵数值show为1
void  MatrixINIT(int ROW, int COL, T* Matrix)
{
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            Matrix[i * COL + j] = (T)(rand() % 100 + 1);
            //Matrix[i * COL + j] = (T)(i * COL + j+1);
        }
    }
}

template <typename T>
//T等0表示不转置,1表示转置。
void Matrixshow(string matrix, int ROW, int COL, T* Matrix, int show = 0, int T_OR_N = 0, string T_ = "None")
{
    if (show) {
        cout << "矩阵" << matrix << ":" << endl << endl;
        for (int i = 0; i < ROW; i++)
        {
            for (int j = 0; j < COL; j++)
            {
                if (T_OR_N)
                {
                    if (T_ == "char") {
                        cout << (int)Matrix[j * ROW + i] << " "; //转置，按行优先顺序读取h_C相当于做了CT的结果
                    }
                    else
                    {
                        cout << (T)Matrix[j * ROW + i] << " "; //转置，按行优先顺序读取h_C相当于做了CT的结果
                    }

                }
                else
                {
                    if (T_ == "char") {
                        cout << (int)Matrix[i * COL + j] << " ";//不转置，按行读取h_C相当于做了CTT=C的结果
                    }
                    else
                    {
                        cout << (T)Matrix[i * COL + j] << " ";//不转置，按行读取h_C相当于做了CTT=C的结果
                    }
                }
            }
            cout << endl;
        }
        cout << endl;
    }

}

template <typename T1, typename T2>
//T等0表示不转置,1表示转置。
void cpu_matrix_mult(T1* h_a, T1* h_b, int m, int n, int k, T2* h_C, T2* h_CC, int T_OR_N = 0)
{
    T2 t;
    TIMER_START(_X);
    //同样计算500次

    for (int kk = 0; kk < N; kk++)
    {
        memset(h_CC, 0, sizeof(T2) * m * n);
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                T2 temp = 0.0f;
                T2 comp = 0.0f;
                for (int h = 0; h < n; ++h)
                {
                    comp -= h_a[i * n + h] * h_b[h * k + j];
                    t = temp - comp;
                    comp = (t - temp) + comp;
                    temp = t;
                }
                h_CC[i * k + j] = temp;
            }
        }
    }
    TIMER_STOP(_X);
    cout << "CPU耗费了: " << TIMER_MSEC(_X) << " ms " << "\n";
  
    bool ok = 1;

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            if (T_OR_N)
            {
                if (fabs(h_CC[i * k + j] - h_C[j * m + i]) > (1.0e-10))
                {
                    ok = 0;
                    cout << h_CC[i * k + j] << " - " << h_C[j * m + i] << " = " << h_CC[i * k + j] - h_C[j * m + i] << "\n";
                }
            }
            else
            {
                if (fabs(h_CC[i * k + j] - h_C[i * k + j]) > (1.0e-10))
                {
                    ok = 0;
                    cout << h_CC[i * k + j] << " - " << h_C[i * k + j] << " = " << h_CC[i * k + j] - h_C[i * k + j] << "\n";
                }
            }
        }
    }
    if (ok)
    {
        cout << "Pass!!!\n \n";
    }
    else
    {
        cout << "Error!!!\n \n";
    }
}
