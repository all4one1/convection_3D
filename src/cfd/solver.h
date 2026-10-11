#pragma once

#include "../common/types.h"
#include "../common/utilities_cuda.h"
#include "../CuPoisson/CuPoisson.h"
#include "solver_ternary.h"
#include <array>

struct Solver 
{
//private:
	Configuration config;
	Arrays d;
	cudaLaunchConfig_t launch1d, launch3d;
	CuPoisson pressure;
	std::array<void*, 5> pressure_args;

public:
	Solver(const Configuration& config_, Arrays& device_)
	{
		d = device_;
		config = config_;

		launch1d = CudaLaunchSetup::set_1d_kernel(config.N, config.nx);
		launch3d = CudaLaunchSetup::set_3d_kernel(config.N, config.nx, config.ny, config.nz);

		pressure_args = { &d.ux, &d.uy, &d.uz, &d.p, &d.p0 };
		pressure = CuPoisson(config.N, Poisson, pressure_args.data(), { 3, 4 }, KERNEL(launch3d));
	};
	
    void proceed_binary()
    {
		quasi_velocity << < KERNEL(launch3d) >> > (d.ux, d.uy, d.uz, d.vx, d.vy, d.vz, d.T, d.T0, d.C, d.C0, d.C2, d.C20);
		temperature << < KERNEL(launch3d) >> > (d.ux, d.uy, d.uz, d.vx, d.vy, d.vz, d.T, d.T0, d.C, d.C0, d.C2, d.C20);
		concentration2 << < KERNEL(launch3d) >> > (d.ux, d.uy, d.uz, d.vx, d.vy, d.vz, d.T, d.T0, d.C, d.C0, d.C2, d.C20);
		pressure.solve_v1b();
		velocity_correction << < KERNEL(launch3d) >> > (d.ux, d.uy, d.uz, d.vx, d.vy, d.vz, d.p);
		swap_6 << < KERNEL(launch1d) >>> (d.ux, d.vx, d.uy, d.vy, d.uz, d.vz, d.T0, d.T, d.C0, d.C, d.C20, d.C2);
    }

	void proceed_binary_fixed_temperature()
	{
		quasi_velocity << < KERNEL(launch3d) >> > (d.ux, d.uy, d.uz, d.vx, d.vy, d.vz, d.T, d.T0, d.C, d.C0, d.C2, d.C20);
		temperature_fixed_value << < KERNEL(launch3d) >> > (d.ux, d.uy, d.uz, d.vx, d.vy, d.vz, d.T, d.T0, d.C, d.C0, d.C2, d.C20);
		concentration2 << < KERNEL(launch3d) >> > (d.ux, d.uy, d.uz, d.vx, d.vy, d.vz, d.T, d.T0, d.C, d.C0, d.C2, d.C20);
		pressure.solve_v1b();
		velocity_correction << < KERNEL(launch3d) >> > (d.ux, d.uy, d.uz, d.vx, d.vy, d.vz, d.p);
		swap_6 << < KERNEL(launch1d) >> > (d.ux, d.vx, d.uy, d.vy, d.uz, d.vz, d.T0, d.T, d.C0, d.C, d.C20, d.C2);
	}

};