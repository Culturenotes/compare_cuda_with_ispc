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

//���Դ���
#define N 500

//�Լ���cuda�˺���
#define USE_MY_INT
//#define USE_MY_FLOAT
//#define USE_MY_DOUBLE

//cublas�ľ������
#define USE_INT8_T
//#define USE_INT8_N
//#define USE_FLOAT_T
//#define USE_FLOAT_N
//#define USE_DOUBLE_T
//#define USE_DOUBLE_N

//ISPC�ľ������
#define USE_ISPC_INT
//#define USE_ISPC_FLOAT
//#define USE_ISPC_DOUBLE

//�Ƿ�ʼ��֤������ȷ��cpu�ļ���ʱ��
#define USE_CPU_COST


//����A��B��C����������
int const A_ROW = 512;
int const A_COL = 512;

int const B_ROW = 512;
int const B_COL = 512;


// ��TIMER_START ����һ��������¼��ʼ��ʱ��
#define TIMER_START(_X) auto _X##_start = std::chrono::system_clock::now(), _X##_stop = _X##_start
// ��TIMER_STOP ����һ��������¼������ʱ��
#define TIMER_STOP(_X) _X##_stop = std::chrono::system_clock::now()
// TIMER_MSEC ����start��stop�������ٺ���
#define TIMER_MSEC(_X) (1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(_X##_stop - _X##_start).count())

template <typename T>
//Ĭ�ϴ�ӡ��ʼ��������ֵshowΪ1
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
//T��0��ʾ��ת��,1��ʾת�á�
void Matrixshow(string matrix, int ROW, int COL, T* Matrix, int show = 0, int T_OR_N = 0, string T_ = "None")
{
    if (show) {
        cout << "����" << matrix << ":" << endl << endl;
        for (int i = 0; i < ROW; i++)
        {
            for (int j = 0; j < COL; j++)
            {
                if (T_OR_N)
                {
                    if (T_ == "char") {
                        cout << (int)Matrix[j * ROW + i] << " "; //ת�ã���������˳���ȡh_C�൱������CT�Ľ��
                    }
                    else
                    {
                        cout << (T)Matrix[j * ROW + i] << " "; //ת�ã���������˳���ȡh_C�൱������CT�Ľ��
                    }

                }
                else
                {
                    if (T_ == "char") {
                        cout << (int)Matrix[i * COL + j] << " ";//��ת�ã����ж�ȡh_C�൱������CTT=C�Ľ��
                    }
                    else
                    {
                        cout << (T)Matrix[i * COL + j] << " ";//��ת�ã����ж�ȡh_C�൱������CTT=C�Ľ��
                    }
                }
            }
            cout << endl;
        }
        cout << endl;
    }

}

template <typename T1, typename T2>
//T��0��ʾ��ת��,1��ʾת�á�
void cpu_matrix_mult(T1* h_a, T1* h_b, int m, int n, int k, T2* h_C, T2* h_CC, int T_OR_N = 0)
{
    T2 t;
    TIMER_START(_X);
    //ͬ������500��

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
    cout << "CPU�ķ���: " << TIMER_MSEC(_X) << " ms " << "\n";
  
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
