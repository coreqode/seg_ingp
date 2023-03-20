#pragma once

#include <neural-graphics-primitives/common.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <vector>

using namespace tcnn;
template <typename T>
void debug_print(const GPUMatrix<T> &device_array, int num_bytes, int n_elements = 32){
    std::vector<float> host_array(num_bytes / sizeof(float));

    CUDA_CHECK_THROW(cudaMemcpy(
        host_array.data(),
        device_array.data(),
        num_bytes,
        cudaMemcpyDeviceToHost
    ));

    int count = 0;
    while (count < n_elements)
    {
        printf("%f ", host_array[count]);
        count++;
    }
    printf("\n--------------------------------\n");
}

template <typename T>
void debug_print(T* device_array, int num_bytes, int n_elements = 32){
	T* host_array = (T*)malloc(num_bytes);

    CUDA_CHECK_THROW(cudaMemcpy(
        host_array,
        device_array,
        num_bytes,
        cudaMemcpyDeviceToHost
    ));

    if (host_array != NULL){
        printf("host_array is not NULL\n");
    }
    else{
        printf("host_array is NULL\n");
    }
    
    if (n_elements > num_bytes / sizeof(T))
        n_elements = num_bytes / sizeof(T);

    int count = 0;
    while (count < n_elements)
    {
        printf("%f ", (float) *(host_array + count));
        count++;
    }
    printf("\n--------------------------------\n");

    free(host_array);
}

/* Checking the values of the rgbsigma matrix 
int num_bytes = rgbsigma_matrix.n_bytes();
std::vector<float> rgbsigma_host(num_bytes);

CUDA_CHECK_THROW(cudaMemcpy(
    rgbsigma_host.data(),
    rgbsigma_matrix.data(),
    num_bytes,
    cudaMemcpyDeviceToHost
));

int count = 0;
while (count < 32)
{
    printf("%f ", rgbsigma_host[count]);
    count++;
}
printf("\n--------------------------------\n");
*/

/* Checking the values of the coords matrix 
int num_bytes = coords_matrix.n_bytes();
std::vector<float> coords_host(num_bytes);

CUDA_CHECK_THROW(cudaMemcpy(
    coords_host.data(),
    coords_matrix.data(),
    num_bytes,
    cudaMemcpyDeviceToHost
));

int count = 0;
while (count < 32)
{
    printf("%f ", coords_host[count]);
    count++;
}
printf("\n--------------------------------\n");
*/