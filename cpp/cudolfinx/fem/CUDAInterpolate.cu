#include <cuda.h>
#include <vector>
#include <array>
#include <concepts>

// Assign A with entries of B as specified in M.
// A and M are of size n.
template<std::floating_point T>
__global__ void _mask_right(T* A, const T* B, const int* M, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[i] = B[M[i]];
    }
}

// Write each entry B[i] to the location M[i] in A.
// B and M are of size n.
template<std::floating_point T>
__global__ void _mask_left(T* A, const T* B, const int* M, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[M[i]] = B[i];
    }
}

// Compute C = AB
// A is m x k, B is k x n, C is m x n
template<std::floating_point T>
__global__ void _matmul(T* C, const T* A, const T* B, int m, int k, int n) {

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if ((row < m) && (col < n)) {
        T s = 0;
        for (int kk = 0; kk < k; kk++) {
            s += A[row*k + kk] * B[kk*n + col];
        }
        C[row*n + col] = s;
    }
}
namespace dolfinx::CUDA {

/// Same-map interpolation kernel.
///
/// @param[out] u1 Pointer to device-side coefficient array that will be updated
/// by the interpolation.
/// @param[in] u0 Pointer to device-side coefficient array to be interpolated FROM
/// @param[in] n1 No. of DOF per element of Function to interpolate INTO
/// @param[in] n0 No. of DOF per element of Function to interpolate FROM
/// @param[in] C No. of cells in the mesh
/// @param[in] bs Element block size 
/// @param[in] i_m Pointer to device-side interpolation matrix with dim n1 x n0
/// @param[in] M1 Pointer to device-side DOF map for u1
/// @param[in] M0 Pointer to device-side DOF map for u0
template<std::floating_point T>
void d_interpolate_same_map(T* u1,
                            T* u0,
                            int n1,
                            int n0, int C,
                            int bs,
                            T* i_m, int* M1, int* M0) {

    T *X0, *X1;
    cudaMalloc((void **)&X0, bs * n0 * C * sizeof(T));
    cudaMalloc((void **)&X1, bs * n1 * C * sizeof(T));

    const int numThreads = 128;
    _mask_right<<<n0 * C * bs / numThreads + 1, numThreads>>>(X0, u0, M0, bs * n0 * C);

    const int matSize = 16;
    dim3 dimGrid(C*bs / matSize + 1, n1 / matSize + 1, 1);
    dim3 dimBlock(matSize, matSize, 1);
    _matmul<<<dimGrid, dimBlock>>>(X1, i_m, X0, n1, n0, C*bs);

    _mask_left<<<n1 * C * bs / numThreads+1 ,numThreads>>>(u1, X1, M1, bs * n1 * C);

    cudaFree(X0);
    cudaFree(X1);
}

    template void d_interpolate_same_map<double>(double*, double*, int, int, int, int, double*, int*, int*);
    template void d_interpolate_same_map<float>(float*, float*, int, int, int, int,float*, int*, int*);
}
