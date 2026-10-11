#pragma once
#include "cuda_runtime.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define cudaCheckError() {cudaError_t e = cudaGetLastError(); \
if (e != cudaSuccess) {printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(0); } }


double get_gpu_memory_used()
{
	size_t free_byte, total_byte;
	cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

	if (cudaSuccess != cuda_status)
	{
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
		return -1;
	}
	double free_mb = (double)free_byte / 1024.0 / 1024.0;
	double total_mb = (double)total_byte / 1024.0 / 1024.0;
	double used_mb = total_mb - free_mb;
	return used_mb;
}

size_t allocate_device_arrays(std::vector<double**> v, size_t N)
{
	size_t Nbytes = sizeof(double) * N;
	size_t total_bytes = 0;
	for (auto it : v)
	{
		cudaMalloc((void**)&(*it), Nbytes);
		cudaMemset(*it, 0, Nbytes);
		total_bytes += Nbytes;
	}
	return total_bytes;
}

struct GPU_
{
	double gpu_mem_allocated = 0;
	double mem_start = 0;
	int devID = 0;

	explicit GPU_(int device_id = 0) : devID(device_id)
	{
		cudaError_t status = cudaSetDevice(devID);
		if (status != cudaSuccess)
			throw std::runtime_error(std::string("cudaSetDevice failed: ") +
				cudaGetErrorString(status));

		mem_start = get_gpu_memory_used();

		cudaDeviceProp deviceProp{};
		status = cudaGetDeviceProperties(&deviceProp, devID);
		if (status != cudaSuccess)
			throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") +
				cudaGetErrorString(status));

		printf("\nDevice %d: \"%s\"\n", devID, deviceProp.name);
	}

	void show_memory_usage_MB()
	{
		std::cout << "Approximately GPU memory allocated: " << gpu_mem_allocated / (1024.0 * 1024) << " MB" << std::endl;
		std::cout << "Approximately GPU memory allocated: " << get_gpu_memory_used() - mem_start << " MB" << std::endl;
	}
};


#define KERNEL1D launch.grid1d, launch.block1d
#define KERNEL2D launch.grid2d, launch.block2d
#define KERNEL3D launch.grid3d, launch.block3d
#define KERNEL(x) (x).gridDim, (x).blockDim 
//, (x).dynamicSmemBytes, (x).stream

struct CudaLaunchSetup
{
	unsigned int threads3d = 8, threads2d = 32, threads = 1024;
	dim3 grid3d, block3d, grid2d, block2d, grid1d, block1d;
	cudaLaunchConfig_t D1, D2, D3;

	CudaLaunchSetup() {};
	CudaLaunchSetup(unsigned int N, unsigned int nx = 1, unsigned int ny = 1, unsigned nz = 1)
	{
		grid3d = dim3(
			(unsigned int)ceil((nx + 1.0) / threads3d),
			(unsigned int)ceil((ny + 1.0) / threads3d),
			(unsigned int)ceil((nz + 1.0) / threads3d));
		block3d = dim3(threads3d, threads3d, threads3d);

		grid2d = dim3(
			(unsigned int)ceil((nx + 1.0) / threads2d),
			(unsigned int)ceil((ny + 1.0) / threads2d));
		block2d = dim3(threads2d, threads2d);

		grid1d = dim3((unsigned int)ceil((N + 0.0) / threads));
		block1d = threads;

		D1.gridDim = grid1d;
		D1.blockDim = block1d;

		D2.gridDim = grid2d;	
		D2.blockDim = block2d;

		D3.gridDim = grid3d;
		D3.blockDim = block3d;
	};

	static cudaLaunchConfig_t set_1d_kernel(unsigned int N, unsigned int nx, unsigned int threads = 1024)
	{
		cudaLaunchConfig_t config{};
		config.gridDim = dim3((unsigned int)ceil((N + 0.0) / threads));
		config.blockDim = threads;
		config.dynamicSmemBytes = 0;
		config.stream = 0;
		return config;
	}
	static cudaLaunchConfig_t set_2d_kernel(unsigned int N, unsigned int nx, unsigned int ny, unsigned int threads2d = 32)
	{
		cudaLaunchConfig_t config{};
		config.gridDim = dim3(
			(unsigned int)ceil((nx + 1.0) / threads2d),
			(unsigned int)ceil((ny + 1.0) / threads2d));
		config.blockDim = dim3(threads2d, threads2d);
		config.dynamicSmemBytes = 0;
		config.stream = 0;
		return config;
	}
	static cudaLaunchConfig_t set_3d_kernel(unsigned int N, unsigned int nx, unsigned int ny, unsigned nz, unsigned int threads3d = 8)
	{
		cudaLaunchConfig_t config{};
		config.gridDim = dim3(
			(unsigned int)ceil((nx + 1.0) / threads3d), 
			(unsigned int)ceil((ny + 1.0) / threads3d), 
			(unsigned int)ceil((nz + 1.0) / threads3d));;
		config.blockDim = dim3(threads3d, threads3d, threads3d);;
		config.dynamicSmemBytes = 0;
		config.stream = 0;
		return config;
	}
};

/**
* dest - vector of destination arrays, one or more
* src - one source array
* N - the size of
*/
void copyArrayFromHostToDevice(std::vector<double*> dest, double* src, unsigned int N)
{
	for (auto& it : dest)
		cudaMemcpy(it, src, N * sizeof(double), cudaMemcpyHostToDevice);
};

