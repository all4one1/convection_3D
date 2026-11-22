#pragma once

__global__ void device_parameter_check() {

	printf("\n thread x:%i y:%i, information copied from device:\n", threadIdx.x, threadIdx.y);
	printf("Ra= %f Pr=%f \n", dev.Ra, dev.Pr);
	printf("psi1= %f psi2= %f psiS= %f\n", dev.psi1, dev.psi2, dev.psiS);
	printf("Sc1= %f Sc2=%f \n", dev.Sc11, dev.Sc22);


	printf("hx= %f hy=%f hz=%f \n", dev.hx, dev.hy, dev.hz);
	printf("tau= %20.16f  \n", dev.tau);
	printf("nx= %i ny=%i nz=%i N=%i \n", dev.nx, dev.ny, dev.nz, dev.N);
	printf("Lx= %f Ly=%f Lz=%f \n", dev.Lx, dev.Ly, dev.Lz);
	printf("offset= %i offset2=%i \n", dev.offset, dev.offset2);
	printf("sinA= %f cosA=%f \n", dev.sinA, dev.cosA);
	printf("xbc= %i ybc=%i zbc=%i \n", (int)dev.xbc, (int)dev.ybc, (int)dev.zbc);

	printf("\n");
}




__device__ __forceinline__ unsigned int periodic_source(unsigned int index, unsigned int last)
{
	if (index == 0) return last - 1;
	if (index == last) return 1;
	return index;
}

__device__ __forceinline__ unsigned int adjacent_interior(unsigned int index, unsigned int last)
{
	if (index == 0) return 1;
	if (index == last) return last - 1;
	return index;
}

__global__ void quasi_velocity(double* ux_d, double* uy_d, double* uz_d,
	double* vx_d, double* vy_d, double* vz_d, double* T_d, double* T0_d,
	double* C1_d, double* C10_d, double* C2_d, double* C20_d)
{
	const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
	const unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
	const unsigned int l = i + dev.offset * j + dev.offset2 * k;
	if (i > dev.nx || j > dev.ny || k > dev.nz || l >= dev.N) return;

	const auto interior_velocity = [&](unsigned int ii, unsigned int jj, unsigned int kk)
		{
			const unsigned int q = ii + dev.offset * jj + dev.offset2 * kk;
			const auto transport = [&](const double* velocity)
				{
					return -vx_d[q] * (velocity[q + 1] - velocity[q - 1]) / 2.0 / dev.hx
						- vy_d[q] * (velocity[q + dev.offset] - velocity[q - dev.offset]) / 2.0 / dev.hy
						- vz_d[q] * (velocity[q + dev.offset2] - velocity[q - dev.offset2]) / 2.0 / dev.hz
						+ (velocity[q + 1] - 2.0 * velocity[q] + velocity[q - 1]) / dev.hx / dev.hx
						+ (velocity[q + dev.offset] - 2.0 * velocity[q] + velocity[q - dev.offset]) / dev.hy / dev.hy
						+ (velocity[q + dev.offset2] - 2.0 * velocity[q] + velocity[q - dev.offset2]) / dev.hz / dev.hz;
				};
			const double buoyancy = dev.Ra / dev.Pr * (T0_d[q] + C10_d[q] + C20_d[q]);
			ux_d[l] = vx_d[q] + dev.tau * (transport(vx_d) + buoyancy * dev.sinA);
			uy_d[l] = vy_d[q] + dev.tau * (transport(vy_d) + buoyancy * dev.cosA);
			uz_d[l] = vz_d[q] + dev.tau * transport(vz_d);
		};

	const unsigned int ii = dev.xbc == periodic ? periodic_source(i, dev.nx) : i;
	const unsigned int kk = dev.zbc == periodic ? periodic_source(k, dev.nz) : k;
	const bool inside_x = ii > 0 && ii < dev.nx;
	const bool inside_y = j > 0 && j < dev.ny;
	const bool inside_z = kk > 0 && kk < dev.nz;
	const unsigned int q = ii + dev.offset * j + dev.offset2 * kk;

	if (inside_x && inside_y && inside_z)
		interior_velocity(ii, j, kk);
	// The pressure equation uses only the normal predictor on solid faces.
	else if (j == 0 && inside_x && inside_z)
		uy_d[l] = dev.tau / dev.hy / dev.hy
		* (2.0 * vy_d[q] - 5.0 * vy_d[q + dev.offset] + 4.0 * vy_d[q + 2 * dev.offset] - vy_d[q + 3 * dev.offset])
		+ dev.tau * dev.Ra / dev.Pr * (T0_d[q] + C10_d[q] + C20_d[q]) * dev.cosA;
	else if (j == dev.ny && inside_x && inside_z)
		uy_d[l] = dev.tau / dev.hy / dev.hy
		* (2.0 * vy_d[q] - 5.0 * vy_d[q - dev.offset] + 4.0 * vy_d[q - 2 * dev.offset] - vy_d[q - 3 * dev.offset])
		+ dev.tau * dev.Ra / dev.Pr * (T0_d[q] + C10_d[q] + C20_d[q]) * dev.cosA;
	else if (dev.xbc == closed && i == 0 && inside_y && inside_z)
		ux_d[l] = dev.tau / dev.hx / dev.hx * (2.0 * vx_d[q] - 5.0 * vx_d[q + 1] + 4.0 * vx_d[q + 2] - vx_d[q + 3])
		+ dev.tau * dev.Ra / dev.Pr * (T0_d[q] + C10_d[q] + C20_d[q]) * dev.sinA;
	else if (dev.xbc == closed && i == dev.nx && inside_y && inside_z)
		ux_d[l] = dev.tau / dev.hx / dev.hx * (2.0 * vx_d[q] - 5.0 * vx_d[q - 1] + 4.0 * vx_d[q - 2] - vx_d[q - 3])
		+ dev.tau * dev.Ra / dev.Pr * (T0_d[q] + C10_d[q] + C20_d[q]) * dev.sinA;
	else if (dev.zbc == closed && k == 0 && inside_x && inside_y)
		uz_d[l] = dev.tau / dev.hz / dev.hz
		* (2.0 * vz_d[q] - 5.0 * vz_d[q + dev.offset2] + 4.0 * vz_d[q + 2 * dev.offset2] - vz_d[q + 3 * dev.offset2]);
	else if (dev.zbc == closed && k == dev.nz && inside_x && inside_y)
		uz_d[l] = dev.tau / dev.hz / dev.hz
		* (2.0 * vz_d[q] - 5.0 * vz_d[q - dev.offset2] + 4.0 * vz_d[q - 2 * dev.offset2] - vz_d[q - 3 * dev.offset2]);
}

__global__ void temperature(double* ux_d, double* uy_d, double* uz_d,
	double* vx_d, double* vy_d, double* vz_d, double* T_d, double* T0_d,
	double* C1_d, double* C10_d, double* C2_d, double* C20_d)
{
	const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
	const unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
	const unsigned int l = i + dev.offset * j + dev.offset2 * k;
	if (i > dev.nx || j > dev.ny || k > dev.nz || l >= dev.N) return;

	const auto interior_temperature = [&](unsigned int ii, unsigned int jj, unsigned int kk)
		{
			const unsigned int q = ii + dev.offset * jj + dev.offset2 * kk;
			const double laplacian = (T0_d[q + 1] - 2.0 * T0_d[q] + T0_d[q - 1]) / dev.hx / dev.hx
				+ (T0_d[q + dev.offset] - 2.0 * T0_d[q] + T0_d[q - dev.offset]) / dev.hy / dev.hy
				+ (T0_d[q + dev.offset2] - 2.0 * T0_d[q] + T0_d[q - dev.offset2]) / dev.hz / dev.hz;
			const double rhs = -vx_d[q] * (T0_d[q + 1] - T0_d[q - 1]) / (2.0 * dev.hx)
				- vy_d[q] * (T0_d[q + dev.offset] - T0_d[q - dev.offset]) / (2.0 * dev.hy)
				- vz_d[q] * (T0_d[q + dev.offset2] - T0_d[q - dev.offset2]) / (2.0 * dev.hz)
				+ laplacian / dev.Pr;
			return T0_d[q] + dev.tau * rhs;
		};

	// Evaluate periodic copies at their partner coordinates.
	const unsigned int ii = dev.xbc == periodic ? periodic_source(i, dev.nx) : i;
	const unsigned int kk = dev.zbc == periodic ? periodic_source(k, dev.nz) : k;
	const bool inside_x = ii > 0 && ii < dev.nx;
	const bool inside_y = j > 0 && j < dev.ny;
	const bool inside_z = kk > 0 && kk < dev.nz;
	const unsigned int q = ii + dev.offset * j + dev.offset2 * kk;

	if (inside_x && inside_y && inside_z)
		T_d[l] = interior_temperature(ii, j, kk);
	else if (inside_x && inside_z)
	{
		const double flux_correction = 2.0 / 3.0 * dev.hy * dev.Q;
		if (j == 0)
			T_d[l] = (4.0 * T0_d[q + dev.offset] - T0_d[q + 2 * dev.offset]) / 3.0 - flux_correction;
		else
			T_d[l] = (4.0 * T0_d[q - dev.offset] - T0_d[q - 2 * dev.offset]) / 3.0 + flux_correction;
	}
	else if (dev.xbc == closed && inside_y && inside_z)
		T_d[l] = i == 0 ? (4.0 * T0_d[q + 1] - T0_d[q + 2]) / 3.0
		: (4.0 * T0_d[q - 1] - T0_d[q - 2]) / 3.0;
	else if (dev.zbc == closed && inside_x && inside_y)
		T_d[l] = k == 0 ? (4.0 * T0_d[q + dev.offset2] - T0_d[q + 2 * dev.offset2]) / 3.0
		: (4.0 * T0_d[q - dev.offset2] - T0_d[q - 2 * dev.offset2]) / 3.0;
	else
	{
		const unsigned int inner = adjacent_interior(ii, dev.nx)
			+ dev.offset * adjacent_interior(j, dev.ny)
			+ dev.offset2 * adjacent_interior(kk, dev.nz);
		T_d[l] = T0_d[inner];
	}
}

__global__ void temperature_fixed_value(double* ux_d, double* uy_d, double* uz_d,
	double* vx_d, double* vy_d, double* vz_d, double* T_d, double* T0_d,
	double* C1_d, double* C10_d, double* C2_d, double* C20_d)
{
	const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
	const unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
	const unsigned int l = i + dev.offset * j + dev.offset2 * k;
	if (i > dev.nx || j > dev.ny || k > dev.nz || l >= dev.N) return;

	const auto interior_temperature = [&](unsigned int ii, unsigned int jj, unsigned int kk)
		{
			const unsigned int q = ii + dev.offset * jj + dev.offset2 * kk;
			const double laplacian = (T0_d[q + 1] - 2.0 * T0_d[q] + T0_d[q - 1]) / dev.hx / dev.hx
				+ (T0_d[q + dev.offset] - 2.0 * T0_d[q] + T0_d[q - dev.offset]) / dev.hy / dev.hy
				+ (T0_d[q + dev.offset2] - 2.0 * T0_d[q] + T0_d[q - dev.offset2]) / dev.hz / dev.hz;
			const double rhs = -vx_d[q] * (T0_d[q + 1] - T0_d[q - 1]) / (2.0 * dev.hx)
				- vy_d[q] * (T0_d[q + dev.offset] - T0_d[q - dev.offset]) / (2.0 * dev.hy)
				- vz_d[q] * (T0_d[q + dev.offset2] - T0_d[q - dev.offset2]) / (2.0 * dev.hz)
				+ laplacian / dev.Pr;
			return T0_d[q] + dev.tau * rhs;
		};

	// Evaluate periodic copies at their partner coordinates.
	const unsigned int ii = dev.xbc == periodic ? periodic_source(i, dev.nx) : i;
	const unsigned int kk = dev.zbc == periodic ? periodic_source(k, dev.nz) : k;
	const bool inside_x = ii > 0 && ii < dev.nx;
	const bool inside_y = j > 0 && j < dev.ny;
	const bool inside_z = kk > 0 && kk < dev.nz;
	const unsigned int q = ii + dev.offset * j + dev.offset2 * kk;

	if (inside_x && inside_y && inside_z)
		T_d[l] = interior_temperature(ii, j, kk);
	else if (inside_x && inside_z)
		T_d[l] = j == 0 ? 1.0 : 0.0;
	else if (dev.xbc == closed && inside_y && inside_z)
		T_d[l] = i == 0 ? (4.0 * T0_d[q + 1] - T0_d[q + 2]) / 3.0
		: (4.0 * T0_d[q - 1] - T0_d[q - 2]) / 3.0;
	else if (dev.zbc == closed && inside_x && inside_y)
		T_d[l] = k == 0 ? (4.0 * T0_d[q + dev.offset2] - T0_d[q + 2 * dev.offset2]) / 3.0
		: (4.0 * T0_d[q - dev.offset2] - T0_d[q - 2 * dev.offset2]) / 3.0;
	else
	{
		const unsigned int inner = adjacent_interior(ii, dev.nx)
			+ dev.offset * adjacent_interior(j, dev.ny)
			+ dev.offset2 * adjacent_interior(kk, dev.nz);
		T_d[l] = T0_d[inner];
	}
}

__global__ void concentration1(double* ux_d, double* uy_d, double* uz_d,
	double* vx_d, double* vy_d, double* vz_d, double* T_d, double* T0_d,
	double* C1_d, double* C10_d, double* C2_d, double* C20_d)
{
	const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
	const unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
	const unsigned int l = i + dev.offset * j + dev.offset2 * k;
	if (i > dev.nx || j > dev.ny || k > dev.nz || l >= dev.N) return;

	const auto interior_concentration = [&](unsigned int ii, unsigned int jj, unsigned int kk)
		{
			const unsigned int q = ii + dev.offset * jj + dev.offset2 * kk;
			const auto laplacian = [&](const double* values)
				{
					return (values[q + 1] - 2.0 * values[q] + values[q - 1]) / dev.hx / dev.hx
						+ (values[q + dev.offset] - 2.0 * values[q] + values[q - dev.offset]) / dev.hy / dev.hy
						+ (values[q + dev.offset2] - 2.0 * values[q] + values[q - dev.offset2]) / dev.hz / dev.hz;
				};
			double rhs = -vx_d[q] * (C10_d[q + 1] - C10_d[q - 1]) / (2.0 * dev.hx)
				- vy_d[q] * (C10_d[q + dev.offset] - C10_d[q - dev.offset]) / (2.0 * dev.hy)
				- vz_d[q] * (C10_d[q + dev.offset2] - C10_d[q - dev.offset2]) / (2.0 * dev.hz)
				+ laplacian(C10_d) / dev.Sc11;
			// Species diffusion acts on C - psi * T.
			rhs -= laplacian(T0_d) / dev.Sc11 * dev.psi1;
			return C10_d[q] + dev.tau * rhs;
		};

	// Evaluate periodic copies at their partner coordinates.
	const unsigned int ii = dev.xbc == periodic ? periodic_source(i, dev.nx) : i;
	const unsigned int kk = dev.zbc == periodic ? periodic_source(k, dev.nz) : k;
	const bool inside_x = ii > 0 && ii < dev.nx;
	const bool inside_y = j > 0 && j < dev.ny;
	const bool inside_z = kk > 0 && kk < dev.nz;
	const unsigned int q = ii + dev.offset * j + dev.offset2 * kk;

	if (inside_x && inside_y && inside_z)
		C1_d[l] = interior_concentration(ii, j, kk);
	else if (inside_x && inside_z)
	{
		const double flux_correction = 2.0 / 3.0 * dev.hy * dev.psi1 * dev.Q;
		if (j == 0)
			C1_d[l] = (4.0 * C10_d[q + dev.offset] - C10_d[q + 2 * dev.offset]) / 3.0 - flux_correction;
		else
			C1_d[l] = (4.0 * C10_d[q - dev.offset] - C10_d[q - 2 * dev.offset]) / 3.0 + flux_correction;
	}
	else if (dev.xbc == closed && inside_y && inside_z)
		C1_d[l] = i == 0 ? (4.0 * C10_d[q + 1] - C10_d[q + 2]) / 3.0
		: (4.0 * C10_d[q - 1] - C10_d[q - 2]) / 3.0;
	else if (dev.zbc == closed && inside_x && inside_y)
		C1_d[l] = k == 0 ? (4.0 * C10_d[q + dev.offset2] - C10_d[q + 2 * dev.offset2]) / 3.0
		: (4.0 * C10_d[q - dev.offset2] - C10_d[q - 2 * dev.offset2]) / 3.0;
	else
	{
		const unsigned int inner = adjacent_interior(ii, dev.nx)
			+ dev.offset * adjacent_interior(j, dev.ny)
			+ dev.offset2 * adjacent_interior(kk, dev.nz);
		C1_d[l] = C10_d[inner];
	}
}

__global__ void concentration2(double* ux_d, double* uy_d, double* uz_d,
	double* vx_d, double* vy_d, double* vz_d, double* T_d, double* T0_d,
	double* C1_d, double* C10_d, double* C2_d, double* C20_d)
{
	const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
	const unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
	const unsigned int l = i + dev.offset * j + dev.offset2 * k;
	if (i > dev.nx || j > dev.ny || k > dev.nz || l >= dev.N) return;

	const auto interior_concentration = [&](unsigned int ii, unsigned int jj, unsigned int kk)
		{
			const unsigned int q = ii + dev.offset * jj + dev.offset2 * kk;
			const auto laplacian = [&](const double* values)
				{
					return (values[q + 1] - 2.0 * values[q] + values[q - 1]) / dev.hx / dev.hx
						+ (values[q + dev.offset] - 2.0 * values[q] + values[q - dev.offset]) / dev.hy / dev.hy
						+ (values[q + dev.offset2] - 2.0 * values[q] + values[q - dev.offset2]) / dev.hz / dev.hz;
				};
			double rhs = -vx_d[q] * (C20_d[q + 1] - C20_d[q - 1]) / (2.0 * dev.hx)
				- vy_d[q] * (C20_d[q + dev.offset] - C20_d[q - dev.offset]) / (2.0 * dev.hy)
				- vz_d[q] * (C20_d[q + dev.offset2] - C20_d[q - dev.offset2]) / (2.0 * dev.hz)
				+ laplacian(C20_d) / dev.Sc22;
			// Species diffusion acts on C - psi * T.
			rhs -= laplacian(T0_d) / dev.Sc22 * dev.psi2;
			return C20_d[q] + dev.tau * rhs;
		};

	// Evaluate periodic copies at their partner coordinates.
	const unsigned int ii = dev.xbc == periodic ? periodic_source(i, dev.nx) : i;
	const unsigned int kk = dev.zbc == periodic ? periodic_source(k, dev.nz) : k;
	const bool inside_x = ii > 0 && ii < dev.nx;
	const bool inside_y = j > 0 && j < dev.ny;
	const bool inside_z = kk > 0 && kk < dev.nz;
	const unsigned int q = ii + dev.offset * j + dev.offset2 * kk;

	if (inside_x && inside_y && inside_z)
		C2_d[l] = interior_concentration(ii, j, kk);
	else if (inside_x && inside_z)
	{
		const double flux_correction = 2.0 / 3.0 * dev.hy * dev.psi2 * dev.Q;
		if (j == 0)
			C2_d[l] = (4.0 * C20_d[q + dev.offset] - C20_d[q + 2 * dev.offset]) / 3.0 - flux_correction;
		else
			C2_d[l] = (4.0 * C20_d[q - dev.offset] - C20_d[q - 2 * dev.offset]) / 3.0 + flux_correction;
	}
	else if (dev.xbc == closed && inside_y && inside_z)
		C2_d[l] = i == 0 ? (4.0 * C20_d[q + 1] - C20_d[q + 2]) / 3.0
		: (4.0 * C20_d[q - 1] - C20_d[q - 2]) / 3.0;
	else if (dev.zbc == closed && inside_x && inside_y)
		C2_d[l] = k == 0 ? (4.0 * C20_d[q + dev.offset2] - C20_d[q + 2 * dev.offset2]) / 3.0
		: (4.0 * C20_d[q - dev.offset2] - C20_d[q - 2 * dev.offset2]) / 3.0;
	else
	{
		const unsigned int inner = adjacent_interior(ii, dev.nx)
			+ dev.offset * adjacent_interior(j, dev.ny)
			+ dev.offset2 * adjacent_interior(kk, dev.nz);
		C2_d[l] = C20_d[inner];
	}
}

__global__ void velocity_correction(double* ux_d, double* uy_d, double* uz_d,
	double* vx_d, double* vy_d, double* vz_d, double* p_d)
{
	const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
	const unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
	const unsigned int l = i + dev.offset * j + dev.offset2 * k;
	if (i > dev.nx || j > dev.ny || k > dev.nz || l >= dev.N) return;

	const auto correct_velocity = [&](unsigned int ii, unsigned int jj, unsigned int kk)
		{
			const unsigned int q = ii + dev.offset * jj + dev.offset2 * kk;
			vx_d[l] = ux_d[q] - dev.tau / (2.0 * dev.hx) * (p_d[q + 1] - p_d[q - 1]);
			vy_d[l] = uy_d[q] - dev.tau / (2.0 * dev.hy) * (p_d[q + dev.offset] - p_d[q - dev.offset]);
			vz_d[l] = uz_d[q] - dev.tau / (2.0 * dev.hz) * (p_d[q + dev.offset2] - p_d[q - dev.offset2]);
		};

	const bool wrap_x = dev.xbc == periodic;
	const bool wrap_z = dev.zbc == periodic;
	const unsigned int ii = wrap_x ? periodic_source(i, dev.nx) : i;
	const unsigned int kk = wrap_z ? periodic_source(k, dev.nz) : k;
	const bool inside_x = ii > 0 && ii < dev.nx;
	const bool inside_y = j > 0 && j < dev.ny;
	const bool inside_z = kk > 0 && kk < dev.nz;

	if ((wrap_x || dev.xbc == closed) && (wrap_z || dev.zbc == closed) && inside_x && inside_y && inside_z)
		correct_velocity(ii, j, kk);
	else
	{
		vx_d[l] = 0.0;
		vy_d[l] = 0.0;
		vz_d[l] = 0.0;
	}
}

__global__ void Poisson(double* ux_d, double* uy_d, double* uz_d, double* p_d, double* p0_d)
{
	const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
	const unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
	const unsigned int l = i + dev.offset * j + dev.offset2 * k;
	if (i > dev.nx || j > dev.ny || k > dev.nz || l >= dev.N) return;

	const bool closed_x = dev.xbc == closed;
	const bool closed_z = dev.zbc == closed;
	const bool wrap_x = dev.xbc == periodic;
	const bool wrap_z = dev.zbc == periodic;
	if ((!closed_x && !wrap_x) || (!closed_z && !wrap_z)) return;
	const auto interior_pressure = [&](unsigned int ii, unsigned int jj, unsigned int kk)
		{
			const unsigned int q = ii + dev.offset * jj + dev.offset2 * kk;
			double correction = -(ux_d[q + 1] - ux_d[q - 1]) / (2.0 * dev.hx * dev.tau)
				- (uy_d[q + dev.offset] - uy_d[q - dev.offset]) / (2.0 * dev.hy * dev.tau)
				- (uz_d[q + dev.offset2] - uz_d[q - dev.offset2]) / (2.0 * dev.hz * dev.tau);

			if (closed_x && ii == 1)
				correction += 2.0 / 3.0 / dev.hx / dev.hx
				* (p0_d[q + 1] - p0_d[q] - dev.hx / dev.tau * ux_d[q - 1]);
			else if (closed_x && ii == dev.nx - 1)
				correction += 2.0 / 3.0 / dev.hx / dev.hx
				* (p0_d[q - 1] - p0_d[q] + dev.hx / dev.tau * ux_d[q + 1]);
			else
				correction += 1.0 / dev.hx / dev.hx * (p0_d[q + 1] + p0_d[q - 1] - 2.0 * p0_d[q]);

			if (jj == 1)
				correction += 2.0 / 3.0 / dev.hy / dev.hy
				* (p0_d[q + dev.offset] - p0_d[q] - dev.hy / dev.tau * uy_d[q - dev.offset]);
			else if (jj == dev.ny - 1)
				correction += 2.0 / 3.0 / dev.hy / dev.hy
				* (p0_d[q - dev.offset] - p0_d[q] + dev.hy / dev.tau * uy_d[q + dev.offset]);
			else
				correction += 1.0 / dev.hy / dev.hy
				* (p0_d[q + dev.offset] + p0_d[q - dev.offset] - 2.0 * p0_d[q]);

			if (closed_z && kk == 1)
				correction += 2.0 / 3.0 / dev.hz / dev.hz
				* (p0_d[q + dev.offset2] - p0_d[q] - dev.hz / dev.tau * uz_d[q - dev.offset2]);
			else if (closed_z && kk == dev.nz - 1)
				correction += 2.0 / 3.0 / dev.hz / dev.hz
				* (p0_d[q - dev.offset2] - p0_d[q] + dev.hz / dev.tau * uz_d[q + dev.offset2]);
			else
				correction += 1.0 / dev.hz / dev.hz
				* (p0_d[q + dev.offset2] + p0_d[q - dev.offset2] - 2.0 * p0_d[q]);

			correction *= dev.tau_p;
			return correction + p0_d[q];
		};

	const unsigned int ii = wrap_x ? periodic_source(i, dev.nx) : i;
	const unsigned int kk = wrap_z ? periodic_source(k, dev.nz) : k;
	const bool inside_x = ii > 0 && ii < dev.nx;
	const bool inside_y = j > 0 && j < dev.ny;
	const bool inside_z = kk > 0 && kk < dev.nz;
	const unsigned int q = ii + dev.offset * j + dev.offset2 * kk;

	if (inside_x && inside_y && inside_z)
		p_d[l] = interior_pressure(ii, j, kk);
	else if (j == 0 && inside_x && inside_z)
		p_d[l] = (4.0 * p0_d[q + dev.offset] - p0_d[q + 2 * dev.offset]) / 3.0
		- uy_d[q] * 2.0 / 3.0 * dev.hy / dev.tau;
	else if (j == dev.ny && inside_x && inside_z)
		p_d[l] = (4.0 * p0_d[q - dev.offset] - p0_d[q - 2 * dev.offset]) / 3.0
		+ uy_d[q] * 2.0 / 3.0 * dev.hy / dev.tau;
	else if (closed_x && i == 0 && inside_y && inside_z)
		p_d[l] = (4.0 * p0_d[q + 1] - p0_d[q + 2]) / 3.0 - ux_d[q] * 2.0 * dev.hx / dev.tau / 3.0;
	else if (closed_x && i == dev.nx && inside_y && inside_z)
		p_d[l] = (4.0 * p0_d[q - 1] - p0_d[q - 2]) / 3.0 + ux_d[q] * 2.0 * dev.hx / dev.tau / 3.0;
	else if (closed_z && k == 0 && inside_x && inside_y)
		p_d[l] = (4.0 * p0_d[q + dev.offset2] - p0_d[q + 2 * dev.offset2]) / 3.0
		- uz_d[q] * 2.0 * dev.hz / dev.tau / 3.0;
	else if (closed_z && k == dev.nz && inside_x && inside_y)
		p_d[l] = (4.0 * p0_d[q - dev.offset2] - p0_d[q - 2 * dev.offset2]) / 3.0
		+ uz_d[q] * 2.0 * dev.hz / dev.tau / 3.0;
	else
	{
		const unsigned int inner = adjacent_interior(ii, dev.nx)
			+ dev.offset * adjacent_interior(j, dev.ny)
			+ dev.offset2 * adjacent_interior(kk, dev.nz);
		p_d[l] = p0_d[inner];
	}
}






__global__ void swap_one(double* f_old, double* f_new) {
	unsigned int l = blockIdx.x * blockDim.x + threadIdx.x;
	if (l < dev.N)	f_old[l] = f_new[l];
}
__global__ void swap_6(double* f1_old, double* f1_new, double* f2_old, double* f2_new, double* f3_old, double* f3_new, double* f4_old, double* f4_new, double* f5_old, double* f5_new, double* f6_old, double* f6_new) {
	unsigned int l = blockIdx.x * blockDim.x + threadIdx.x;
	if (l < dev.N)
	{
		f1_old[l] = f1_new[l];
		f2_old[l] = f2_new[l];
		f3_old[l] = f3_new[l];
		f4_old[l] = f4_new[l];
		f5_old[l] = f5_new[l];
		f6_old[l] = f6_new[l];
	}
}

