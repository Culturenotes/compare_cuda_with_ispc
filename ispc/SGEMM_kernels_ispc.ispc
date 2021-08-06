/*
  Copyright (c) 2018, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Various ISPC SGEMM kernel and task/kernel implementations
// Junkins, September 2018

#define TILE_SIZE 32

export uniform int SGEMM_get_program_count() {
    return programCount;
}

export uniform int SGEMM_get_tile_size() {
    return TILE_SIZE;
}

// This version is modified version of 'SGEMM_tileNoSIMDIntrin'.
// The tile used to read/write values for re-use is a 2D block of height 2 instead of a n array of same width.
#define TILE_HEIGHT 2
#define TILE_WIDTH 32

// This version is a further modified version of 'SGEMM_tileBlockNoSIMDIntrin'.
// Since we already know the tile height, the loop used to access the tile vertically is replaced.
export void SGEMM_tileBlockNoSIMDIntrin_2_int(uniform int matrixA[], uniform int matrixB[], uniform int matrixC[],
    uniform int M, uniform int N, uniform int K) {
    uniform int sumTile[TILE_HEIGHT][TILE_WIDTH];
    uniform int oneAVal[TILE_HEIGHT];

    for (uniform unsigned int m = 0; m < M; m += TILE_HEIGHT) {
        for (uniform unsigned int k0 = 0; k0 < K; k0 += TILE_WIDTH) {
            // SPMD "horizontally" over TILE dimension.
            foreach (ki = 0 ... TILE_WIDTH) {
                // No scatter required.
                sumTile[0][ki] = 0;
                sumTile[1][ki] = 0;
            }

            // Loop over the the matrix N dimension:
            for (uniform unsigned int n = 0; n < N; n++) {
                uniform unsigned int mTimesNPlusN = (m + 0)*N + n;
                uniform unsigned int mPlusOneTimesNPlusN = (m + 1)*N + n;
                prefetch_nt(&matrixA[mPlusOneTimesNPlusN]);
                prefetch_nt(&matrixA[mTimesNPlusN]);

                oneAVal[0] = matrixA[mTimesNPlusN];
                oneAVal[1] = matrixA[mPlusOneTimesNPlusN];

                uniform unsigned int nTimesKPlusk0 = n*K + k0;
                // SPMD iterate over the TILE dimension, but within for loop nest:
                foreach (kt = 0 ... TILE_WIDTH) {
                    // Note, no gather required.
                    varying int matB1 = matrixB[nTimesKPlusk0 + kt];
                    // Pure SIMD FMAC:
                    sumTile[0][kt] += oneAVal[0] * matB1;
                    sumTile[1][kt] += oneAVal[1] * matB1;
                }
            }
            uniform unsigned int mTimesKPlusK0 = (m + 0)*K + k0;
            uniform unsigned int mPlusOneTimesKPlusK0 = (m + 1)*K + k0;
            // SPMD "horizontally" again over TILE dimension:
            foreach (ki = 0 ... TILE_WIDTH) {
                matrixC[mTimesKPlusK0 + ki] = sumTile[0][ki];
                matrixC[mPlusOneTimesKPlusK0 + ki] = sumTile[1][ki];
            }
        }
    }
}

// Multiple task version of the above:
task void SGEMM_tileBlockNoSIMDIntrin_2_task_int(uniform int matrixA[], uniform int matrixB[], uniform int matrixC[],
    uniform int M, uniform int N, uniform int K) {
    uniform int sumTile[TILE_HEIGHT][TILE_WIDTH];
    uniform int oneAVal[TILE_HEIGHT];

    // Determine workset for this task instance:
    uniform unsigned int uNumRowsPerTask = M / taskCount;
    uniform unsigned int uRowStart = uNumRowsPerTask * taskIndex;
    uniform unsigned int uRowEnd = uRowStart + uNumRowsPerTask;

    for (uniform unsigned int m = uRowStart; m < uRowEnd; m += TILE_HEIGHT) {
        for (uniform unsigned int k0 = 0; k0 < K; k0 += TILE_WIDTH) {
            // SPMD "horizontally" over TILE dimension:
            foreach (ki = 0 ... TILE_WIDTH) {
                // No scatter required.
                sumTile[0][ki] = 0;
                sumTile[1][ki] = 0;
            }

            // Loop over the the matrix N dimension:
            for (uniform unsigned int n = 0; n < N; n++) {
                uniform unsigned int mTimesNPlusN = (m + 0)*N + n;
                uniform unsigned int mPlusOneTimesNPlusN = (m + 1)*N + n;
                prefetch_nt(&matrixA[mPlusOneTimesNPlusN]);
                prefetch_nt(&matrixA[mTimesNPlusN]);

                oneAVal[0] = matrixA[mTimesNPlusN];
                oneAVal[1] = matrixA[mPlusOneTimesNPlusN];

                uniform unsigned int nTimesKPlusk0 = n*K + k0;
                // SPMD iterate over the TILE dimension, but within for loop nest:
                foreach (kt = 0 ... TILE_WIDTH) {
                    // Note, no gather required.
                    varying int matB1 = matrixB[nTimesKPlusk0 + kt];
                    // Pure SIMD FMAC:
                    sumTile[0][kt] += oneAVal[0] * matB1;
                    sumTile[1][kt] += oneAVal[1] * matB1;
                }
            }
            uniform unsigned int mTimesKPlusK0 = (m + 0)*K + k0;
            uniform unsigned int mPlusOneTimesKPlusK0 = (m + 1)*K + k0;
            // SPMD "horizontally" again over TILE dimension:
            foreach (ki = 0 ... TILE_WIDTH) {
                matrixC[mTimesKPlusK0 + ki] = sumTile[0][ki];
                matrixC[mPlusOneTimesKPlusK0 + ki] = sumTile[1][ki];
            }
        }
    }
}

export void SGEMM_tileBlockNoSIMDIntrin_2_withTasks_int(uniform int matA[], uniform int matB[], uniform int matC[],
    uniform int M, uniform int N, uniform int K) {
    // The algorithm divides rows in matrix C (M size) between tasks.
    // We want each task to process programCount rows in C matrix to maximize SIMD usage.
    uniform int numTasks = M / programCount;
    launch[numTasks] SGEMM_tileBlockNoSIMDIntrin_2_task_int(matA, matB, matC, M, N, K);
}



// This version is a further modified version of 'SGEMM_tileBlockNoSIMDIntrin'.
// Since we already know the tile height, the loop used to access the tile vertically is replaced.
export void SGEMM_tileBlockNoSIMDIntrin_2_float(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[],
    uniform int M, uniform int N, uniform int K) {
    uniform float sumTile[TILE_HEIGHT][TILE_WIDTH];
    uniform float oneAVal[TILE_HEIGHT];

    for (uniform unsigned int m = 0; m < M; m += TILE_HEIGHT) {
        for (uniform unsigned int k0 = 0; k0 < K; k0 += TILE_WIDTH) {
            // SPMD "horizontally" over TILE dimension.
            foreach (ki = 0 ... TILE_WIDTH) {
                // No scatter required.
                sumTile[0][ki] = 0.0f;
                sumTile[1][ki] = 0.0f;
            }

            // Loop over the the matrix N dimension:
            for (uniform unsigned int n = 0; n < N; n++) {
                uniform unsigned int mTimesNPlusN = (m + 0)*N + n;
                uniform unsigned int mPlusOneTimesNPlusN = (m + 1)*N + n;
                prefetch_nt(&matrixA[mPlusOneTimesNPlusN]);
                prefetch_nt(&matrixA[mTimesNPlusN]);

                oneAVal[0] = matrixA[mTimesNPlusN];
                oneAVal[1] = matrixA[mPlusOneTimesNPlusN];

                uniform unsigned int nTimesKPlusk0 = n*K + k0;
                // SPMD iterate over the TILE dimension, but within for loop nest:
                foreach (kt = 0 ... TILE_WIDTH) {
                    // Note, no gather required.
                    varying float matB1 = matrixB[nTimesKPlusk0 + kt];
                    // Pure SIMD FMAC:
                    sumTile[0][kt] += oneAVal[0] * matB1;
                    sumTile[1][kt] += oneAVal[1] * matB1;
                }
            }
            uniform unsigned int mTimesKPlusK0 = (m + 0)*K + k0;
            uniform unsigned int mPlusOneTimesKPlusK0 = (m + 1)*K + k0;
            // SPMD "horizontally" again over TILE dimension:
            foreach (ki = 0 ... TILE_WIDTH) {
                matrixC[mTimesKPlusK0 + ki] = sumTile[0][ki];
                matrixC[mPlusOneTimesKPlusK0 + ki] = sumTile[1][ki];
            }
        }
    }
}

// Multiple task version of the above:
task void SGEMM_tileBlockNoSIMDIntrin_2_task_float(uniform float matrixA[], uniform float matrixB[], uniform float matrixC[],
    uniform int M, uniform int N, uniform int K) {
    uniform float sumTile[TILE_HEIGHT][TILE_WIDTH];
    uniform float oneAVal[TILE_HEIGHT];

    // Determine workset for this task instance:
    uniform unsigned int uNumRowsPerTask = M / taskCount;
    uniform unsigned int uRowStart = uNumRowsPerTask * taskIndex;
    uniform unsigned int uRowEnd = uRowStart + uNumRowsPerTask;

    for (uniform unsigned int m = uRowStart; m < uRowEnd; m += TILE_HEIGHT) {
        for (uniform unsigned int k0 = 0; k0 < K; k0 += TILE_WIDTH) {
            // SPMD "horizontally" over TILE dimension:
            foreach (ki = 0 ... TILE_WIDTH) {
                // No scatter required.
                sumTile[0][ki] = 0.0f;
                sumTile[1][ki] = 0.0f;
            }

            // Loop over the the matrix N dimension:
            for (uniform unsigned int n = 0; n < N; n++) {
                uniform unsigned int mTimesNPlusN = (m + 0)*N + n;
                uniform unsigned int mPlusOneTimesNPlusN = (m + 1)*N + n;
                prefetch_nt(&matrixA[mPlusOneTimesNPlusN]);
                prefetch_nt(&matrixA[mTimesNPlusN]);

                oneAVal[0] = matrixA[mTimesNPlusN];
                oneAVal[1] = matrixA[mPlusOneTimesNPlusN];

                uniform unsigned int nTimesKPlusk0 = n*K + k0;
                // SPMD iterate over the TILE dimension, but within for loop nest:
                foreach (kt = 0 ... TILE_WIDTH) {
                    // Note, no gather required.
                    varying float matB1 = matrixB[nTimesKPlusk0 + kt];
                    // Pure SIMD FMAC:
                    sumTile[0][kt] += oneAVal[0] * matB1;
                    sumTile[1][kt] += oneAVal[1] * matB1;
                }
            }
            uniform unsigned int mTimesKPlusK0 = (m + 0)*K + k0;
            uniform unsigned int mPlusOneTimesKPlusK0 = (m + 1)*K + k0;
            // SPMD "horizontally" again over TILE dimension:
            foreach (ki = 0 ... TILE_WIDTH) {
                matrixC[mTimesKPlusK0 + ki] = sumTile[0][ki];
                matrixC[mPlusOneTimesKPlusK0 + ki] = sumTile[1][ki];
            }
        }
    }
}

export void SGEMM_tileBlockNoSIMDIntrin_2_withTasks_float(uniform float matA[], uniform float matB[], uniform float matC[],
    uniform int M, uniform int N, uniform int K) {
    // The algorithm divides rows in matrix C (M size) between tasks.
    // We want each task to process programCount rows in C matrix to maximize SIMD usage.
    uniform int numTasks = M / programCount;
    launch[numTasks] SGEMM_tileBlockNoSIMDIntrin_2_task_float(matA, matB, matC, M, N, K);
}



// This version is a further modified version of 'SGEMM_tileBlockNoSIMDIntrin'.
// Since we already know the tile height, the loop used to access the tile vertically is replaced.
export void SGEMM_tileBlockNoSIMDIntrin_2_double(uniform double matrixA[], uniform double matrixB[], uniform double matrixC[],
    uniform int M, uniform int N, uniform int K) {
    uniform double sumTile[TILE_HEIGHT][TILE_WIDTH];
    uniform double oneAVal[TILE_HEIGHT];

    for (uniform unsigned int m = 0; m < M; m += TILE_HEIGHT) {
        for (uniform unsigned int k0 = 0; k0 < K; k0 += TILE_WIDTH) {
            // SPMD "horizontally" over TILE dimension.
            foreach (ki = 0 ... TILE_WIDTH) {
                // No scatter required.
                sumTile[0][ki] = 0.0d;
                sumTile[1][ki] = 0.0d;
            }

            // Loop over the the matrix N dimension:
            for (uniform unsigned int n = 0; n < N; n++) {
                uniform unsigned int mTimesNPlusN = (m + 0)*N + n;
                uniform unsigned int mPlusOneTimesNPlusN = (m + 1)*N + n;
                prefetch_nt(&matrixA[mPlusOneTimesNPlusN]);
                prefetch_nt(&matrixA[mTimesNPlusN]);

                oneAVal[0] = matrixA[mTimesNPlusN];
                oneAVal[1] = matrixA[mPlusOneTimesNPlusN];

                uniform unsigned int nTimesKPlusk0 = n*K + k0;
                // SPMD iterate over the TILE dimension, but within for loop nest:
                foreach (kt = 0 ... TILE_WIDTH) {
                    // Note, no gather required.
                    varying double matB1 = matrixB[nTimesKPlusk0 + kt];
                    // Pure SIMD FMAC:
                    sumTile[0][kt] += oneAVal[0] * matB1;
                    sumTile[1][kt] += oneAVal[1] * matB1;
                }
            }
            uniform unsigned int mTimesKPlusK0 = (m + 0)*K + k0;
            uniform unsigned int mPlusOneTimesKPlusK0 = (m + 1)*K + k0;
            // SPMD "horizontally" again over TILE dimension:
            foreach (ki = 0 ... TILE_WIDTH) {
                matrixC[mTimesKPlusK0 + ki] = sumTile[0][ki];
                matrixC[mPlusOneTimesKPlusK0 + ki] = sumTile[1][ki];
            }
        }
    }
}

// Multiple task version of the above:
task void SGEMM_tileBlockNoSIMDIntrin_2_task_double(uniform double matrixA[], uniform double matrixB[], uniform double matrixC[],
    uniform int M, uniform int N, uniform int K) {
    uniform double sumTile[TILE_HEIGHT][TILE_WIDTH];
    uniform double oneAVal[TILE_HEIGHT];

    // Determine workset for this task instance:
    uniform unsigned int uNumRowsPerTask = M / taskCount;
    uniform unsigned int uRowStart = uNumRowsPerTask * taskIndex;
    uniform unsigned int uRowEnd = uRowStart + uNumRowsPerTask;

    for (uniform unsigned int m = uRowStart; m < uRowEnd; m += TILE_HEIGHT) {
        for (uniform unsigned int k0 = 0; k0 < K; k0 += TILE_WIDTH) {
            // SPMD "horizontally" over TILE dimension:
            foreach (ki = 0 ... TILE_WIDTH) {
                // No scatter required.
                sumTile[0][ki] = 0.0d;
                sumTile[1][ki] = 0.0d;
            }

            // Loop over the the matrix N dimension:
            for (uniform unsigned int n = 0; n < N; n++) {
                uniform unsigned int mTimesNPlusN = (m + 0)*N + n;
                uniform unsigned int mPlusOneTimesNPlusN = (m + 1)*N + n;
                prefetch_nt(&matrixA[mPlusOneTimesNPlusN]);
                prefetch_nt(&matrixA[mTimesNPlusN]);

                oneAVal[0] = matrixA[mTimesNPlusN];
                oneAVal[1] = matrixA[mPlusOneTimesNPlusN];

                uniform unsigned int nTimesKPlusk0 = n*K + k0;
                // SPMD iterate over the TILE dimension, but within for loop nest:
                foreach (kt = 0 ... TILE_WIDTH) {
                    // Note, no gather required.
                    varying double matB1 = matrixB[nTimesKPlusk0 + kt];
                    // Pure SIMD FMAC:
                    sumTile[0][kt] += oneAVal[0] * matB1;
                    sumTile[1][kt] += oneAVal[1] * matB1;
                }
            }
            uniform unsigned int mTimesKPlusK0 = (m + 0)*K + k0;
            uniform unsigned int mPlusOneTimesKPlusK0 = (m + 1)*K + k0;
            // SPMD "horizontally" again over TILE dimension:
            foreach (ki = 0 ... TILE_WIDTH) {
                matrixC[mTimesKPlusK0 + ki] = sumTile[0][ki];
                matrixC[mPlusOneTimesKPlusK0 + ki] = sumTile[1][ki];
            }
        }
    }
}

export void SGEMM_tileBlockNoSIMDIntrin_2_withTasks_double(uniform double matA[], uniform double matB[], uniform double matC[],
    uniform int M, uniform int N, uniform int K) {
    // The algorithm divides rows in matrix C (M size) between tasks.
    // We want each task to process programCount rows in C matrix to maximize SIMD usage.
    uniform int numTasks = M / programCount;
    launch[numTasks] SGEMM_tileBlockNoSIMDIntrin_2_task_double(matA, matB, matC, M, N, K);
}
