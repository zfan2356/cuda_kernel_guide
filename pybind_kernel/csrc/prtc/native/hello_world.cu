#include <bit.h>
#include <cuda_runtime_api.h>
#include <iostream>

__device__ float devData;

__global__ void checkGlobalVar() {
  printf("Device: The value of the global variable is %f\n", devData);
  devData += 2.0;
}

int main() {
  float val = 3.14f;
  cudaMemcpyToSymbol(&devData, &val, sizeof(float));
  std::cout << "Host: copy" << std::endl;

  checkGlobalVar<<<1, 1>>>();

  cudaMemcpyFromSymbol(&val, &devData, sizeof(float));
  std::cout << val << std::endl;
  cudaDeviceReset();
  return 0;
}
