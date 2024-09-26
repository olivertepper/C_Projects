#include <stdio.h>

// Kernel function that runs on the GPU
__global__ void helloFromGPU() {
    printf("Hello from the GPU!\n");
}

int main() {
    // Query GPU properties
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found!\n");
        return 1;
    }

    // Get and print the GPU properties of the first available device
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Device name: %s\n", deviceProp.name);
    printf("CUDA Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Total Global Memory: %lu MB\n", deviceProp.totalGlobalMem / (1024 * 1024));

    // Launch a kernel with 1 block and 1 thread
    helloFromGPU<<<1, 1>>>();

    // Wait for the GPU to finish before accessing the result
    cudaDeviceSynchronize();

    return 0;
}

