#include "cuda_runtime.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

template <typename T, int Size> struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];
  __host__ __device__ inline const T &operator[](int i) const { return val[i]; }
  __host__ __device__ inline T &operator[](int i) { return val[i]; }
};

__device__ float TanhApprox(float x) {
  /*
  ptx:
  float r;
  asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
  return  r;
  */
  return tanhf(x); // cuda inline math API
}

// gelu: x / 2 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
template <typename T> struct GeluFunctor {
  static constexpr T alpha = static_cast<T>(0.7978845608028654); // sqrt(2 / pi)
  static constexpr T beta = static_cast<T>(0.044714998453855515); // 0.044715

  __device__ GeluFunctor() {};

  __device__ T operator()(T x) const {
    const T half = static_cast<T>(0.5);
    const T one = static_cast<T>(1.0);
    const T tanh_in = alpha * (x + beta * x * x * x);
    return half * x * (one + tanh(tanh_in));
  }
};

template <> struct GeluFunctor<half> {
  static constexpr float alpha = GeluFunctor<float>::alpha;
  static constexpr float beta = GeluFunctor<float>::beta;
  GeluFunctor<float> float_functor;

  __device__ GeluFunctor() {}

  __device__ __half operator()(const __half x) const {
    return static_cast<__half>(float_functor(static_cast<float>(x)));
  }
};

template <int VecSize>
__global__ void FP16GeluCUDAKernel(const __half *x, __half *y, int n) {
  int offset =
      static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) * VecSize;
  int stride = static_cast<int>(blockDim.x * gridDim.x) * VecSize;

  GeluFunctor<half> gelu_fwd;
  half y_reg[VecSize];

  using ArrT = AlignedVector<half, VecSize>;

  for (; offset < n; offset += stride) {
    const half *in = x + offset;

    if (VecSize == 1) {
      y_reg[0] = gelu_fwd(in[0]);
    } else {
      for (int i = 0; i < VecSize; i++) {
        y_reg[i] = gelu_fwd(in[i]);
      }
    }
  }
  *reinterpret_cast<ArrT *>(y + offset) = *reinterpret_cast<ArrT *>(y_reg);
}

int main() {
  constexpr int N = 1000;

  // __half is a low-level data type provided by CUDA explicitly for
  // representing 16-bit floating-point numbers.
  // __half operations performed using this type can lead to better performance
  // than half
  __half *x = new __half[N];
  __half *y = new __half[N];

  for (int i = 0; i < N; i++) {
    x[i] = (__half)(i);
  }

  __half *d_x, *d_y;
  cudaMalloc((void **)&d_x, N * sizeof(__half));
  cudaMalloc((void **)&d_y, N * sizeof(__half));
  cudaMemcpy(d_x, x, sizeof(__half) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, sizeof(__half) * N, cudaMemcpyHostToDevice);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  // 检查内存地址对齐的lambda函数
  auto is_aligned = [](const void *ptr, int alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
  };

  constexpr auto kAlignment = alignof(AlignedVector<__half, 8>);

  if (N % 8 == 0 && is_aligned(x, kAlignment) && is_aligned(y, kAlignment)) {
    int thread = std::min(512, deviceProp.maxThreadsPerBlock);
    int block = (N + thread - 1) / thread;

    block = std::min(block, deviceProp.maxGridSize[0]);
    FP16GeluCUDAKernel<1><<<block, thread>>>(d_x, d_y, N);
    cudaMemcpy(y, d_y, sizeof(__half) * N, cudaMemcpyDeviceToHost);
  }

  printf("pass\n");
  delete x;
  x = nullptr;

  delete y;
  y = nullptr;

  cudaFree(d_x);
  cudaFree(d_y);

  return 0;
}
