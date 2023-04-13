#pragma once

#include <neural-graphics-primitives/common.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <vector>
#include <stdio.h>

using namespace tcnn;
template <typename T>
void debug_print(const GPUMatrix<T> &device_array, int num_bytes, int start, int stride, int n_elements){
    std::vector<T> host_array(num_bytes / sizeof(T));

    CUDA_CHECK_THROW(cudaMemcpy(
        host_array.data(),
        device_array.data(),
        num_bytes,
        cudaMemcpyDeviceToHost
    ));

    int count=0;
    while (count < n_elements)
    {
        printf("%f ", (float) host_array[start + count*stride]);
        count++;
    }
    printf("\n--------------------------------\n");
}

template <typename T>
void debug_print(const GPUMatrixDynamic<T> &device_array, int num_bytes, int start, int stride, int n_elements){
    std::vector<T> host_array(num_bytes / sizeof(T));

    CUDA_CHECK_THROW(cudaMemcpy(
        host_array.data(),
        device_array.data(),
        num_bytes,
        cudaMemcpyDeviceToHost
    ));

    int count=0;
    while (count < n_elements)
    {
        printf("%f ", (float) host_array[start + count*stride]);
        count++;
    }
    printf("\n--------------------------------\n");
}



template <typename T>
void debug_print_ptr(T* device_array, int num_bytes, int stride =1, int n_elements = 0){
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

    if (n_elements == 0) {
        n_elements = num_bytes / sizeof(T);
    }

    int count = 0;
    while (count < n_elements)
    {
        printf("%f ", (float) *(host_array + count*stride));
        count++;
    }
    printf("\n--------------------------------\n");

    free(host_array);
}


template <typename T>
void debug_info(const GPUMatrix<T> &matrix){
    printf("\n matrix dimensions: %d x %d \n", matrix.rows(), matrix.cols());
    printf("\n matrix layout RM/CM: %d \n", matrix.layout() == tcnn::RM ? 1 : 0);
    printf("\n matrix layout AoS/SoA: %d \n", matrix.layout() == tcnn::AoS ? 1 : 0);
}

template <typename T>
void debug_info(const GPUMatrixDynamic<T> &matrix){
    printf("\n matrix dimensions: %d x %d ", matrix.rows(), matrix.cols());
    printf("\n matrix layout RM/CM: %d ", matrix.layout() == tcnn::RM ? 1 : 0);
    printf("\n matrix layout AoS/SoA: %d ", matrix.layout() == tcnn::AoS ? 1 : 0);
    printf("\n--------------------------------");
}
