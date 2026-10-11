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




__global__ void quasi_velocity(double* ux_d, double* uy_d, double* uz_d, double* vx_d, double* vy_d, double* vz_d, double* T_d, double* T0_d, double* C1_d, double* C10_d, double* C2_d, double* C20_d) {

	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
	unsigned int l = i + dev.offset * j + dev.offset2 * k;
	if (i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N)
	{

		/*
		INNER
		*/
		if (i > 0 && i < dev.nx && j > 0 && j < dev.ny && k > 0 && k < dev.nz)
		{

			//ux_d
			ux_d[l] = vx_d[l]
				+ dev.tau * (
					-vx_d[l] * (vx_d[l + 1] - vx_d[l - 1]) / 2.0 / dev.hx					//  (ux_d*d/dx) ux_d 
					- vy_d[l] * (vx_d[l + dev.offset] - vx_d[l - dev.offset]) / 2.0 / dev.hy				//  (uy_d*d/dy) ux_d 
					- vz_d[l] * (vx_d[l + dev.offset2] - vx_d[l - dev.offset2]) / 2.0 / dev.hz				//  (uy_d*d/dy) ux_d 

					+ (vx_d[l + 1] - 2.0 * vx_d[l] + vx_d[l - 1]) / dev.hx / dev.hx				// (d2/dx2) ux_d
					+ (vx_d[l + dev.offset] - 2.0 * vx_d[l] + vx_d[l - dev.offset]) / dev.hy / dev.hy			// (d2/dy2) ux_d   
					+ (vx_d[l + dev.offset2] - 2.0 * vx_d[l] + vx_d[l - dev.offset2]) / dev.hz / dev.hz			// (d2/dz2) ux_d 

					+ dev.Ra / dev.Pr * (T0_d[l] + C10_d[l] + C20_d[l]) * dev.sinA
					);

			//uy_d
			uy_d[l] = vy_d[l]
				+ dev.tau * (
					-vx_d[l] * (vy_d[l + 1] - vy_d[l - 1]) / 2.0 / dev.hx					//  (ux_d*d/dx) uy_d
					- vy_d[l] * (vy_d[l + dev.offset] - vy_d[l - dev.offset]) / 2.0 / dev.hy               //  (uy_d*d/dy) uy_d  
					- vz_d[l] * (vy_d[l + dev.offset2] - vy_d[l - dev.offset2]) / 2.0 / dev.hz				//  (uy_d*d/dy) uy_d 

					+ (vy_d[l + 1] - 2.0 * vy_d[l] + vy_d[l - 1]) / dev.hx / dev.hx				// (d2/dx2) uy_d
					+ (vy_d[l + dev.offset] - 2.0 * vy_d[l] + vy_d[l - dev.offset]) / dev.hy / dev.hy			// (d2/dy2) uy_d   
					+ (vy_d[l + dev.offset2] - 2.0 * vy_d[l] + vy_d[l - dev.offset2]) / dev.hz / dev.hz			// (d2/dz2) uy_d 

					+ dev.Ra / dev.Pr * (T0_d[l] + C10_d[l] + C20_d[l]) * dev.cosA 						 // Ra/Pe*T_d
					);

			//uz_d
			uz_d[l] = vz_d[l]
				+ dev.tau * (
					-vx_d[l] * (vz_d[l + 1] - vz_d[l - 1]) / 2.0 / dev.hx					//  (ux_d*d/dx) uz_d 
					- vy_d[l] * (vz_d[l + dev.offset] - vz_d[l - dev.offset]) / 2.0 / dev.hy				//  (uy_d*d/dy) uz_d 
					- vz_d[l] * (vz_d[l + dev.offset2] - vz_d[l - dev.offset2]) / 2.0 / dev.hz				//  (uy_d*d/dy) uz_d 

					+ (vz_d[l + 1] - 2.0 * vz_d[l] + vz_d[l - 1]) / dev.hx / dev.hx				// (d2/dx2) uz_d
					+ (vz_d[l + dev.offset] - 2.0 * vz_d[l] + vz_d[l - dev.offset]) / dev.hy / dev.hy			// (d2/dy2) uz_d   
					+ (vz_d[l + dev.offset2] - 2.0 * vz_d[l] + vz_d[l - dev.offset2]) / dev.hz / dev.hz			// (d2/dz2) uz_d 

					);

		}

		/*
		UP-DOWN
		*/

		// y
		else if (j == 0 && (i > 0 && i < dev.nx && k > 0 && k < dev.nz))
		{
			uy_d[l] = dev.tau / dev.hy / dev.hy * (2.0 * vy_d[l] - 5.0 * vy_d[l + dev.offset] + 4.0 * vy_d[l + dev.offset * 2] - vy_d[l + dev.offset * 3]) + dev.tau * dev.Ra / dev.Pr * (T0_d[l] + C10_d[l] + C20_d[l]) * dev.cosA;
			//ux_d[l] = dev.tau / dev.hy / dev.hy*(2 * vx_d[l] - 5 * vx_d[l + dev.offset] + 4 * vx_d[l + dev.offset * 2] - vx_d[l + dev.offset * 3]);
			//uz_d[l] = dev.tau / dev.hz / dev.hz*(2 * vz_d[l] - 5 * vz_d[l + dev.offset] + 4 * vz_d[l + dev.offset * 2] - vz_d[l + dev.offset * 3]);
		}
		else if (j == dev.ny && (i > 0 && i < dev.nx && k > 0 && k < dev.nz))
		{
			uy_d[l] = dev.tau / dev.hy / dev.hy * (2.0 * vy_d[l] - 5.0 * vy_d[l - dev.offset] + 4.0 * vy_d[l - dev.offset * 2] - vy_d[l - dev.offset * 3]) + dev.tau * dev.Ra / dev.Pr * (T0_d[l] + C10_d[l] + C20_d[l]) * dev.cosA;
			//ux_d[l] = dev.tau / dev.hy / dev.hy*(2 * vx_d[l] - 5 * vx_d[l - dev.offset] + 4 * vx_d[l - dev.offset * 2] - vx_d[l - dev.offset * 3]);
			//uz_d[l] = dev.tau / dev.hy / dev.hy*(2 * vz_d[l] - 5 * vz_d[l - dev.offset] + 4 * vz_d[l - dev.offset * 2] - vz_d[l - dev.offset * 3]);
		}

		/*
		CLOSED
		*/

		// x
		else if (dev.xbc == closed && i == 0 && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))
		{
			ux_d[l] = dev.tau / dev.hx / dev.hx * (2.0 * vx_d[l] - 5.0 * vx_d[l + 1] + 4.0 * vx_d[l + 2] - vx_d[l + 3]) + dev.tau * dev.Ra / dev.Pr * (T0_d[l] + C10_d[l] + C20_d[l]) * dev.sinA;
			//uy_d[l] = dev.tau / dev.hx / dev.hx * (2 * vy_d[l] - 5 * vy_d[l + 1] + 4 * vy_d[l + 2] - vy_d[l + 3]);
			//uz_d[l] = dev.tau / dev.hx / dev.hx * (2 * vz_d[l] - 5 * vz_d[l + 1] + 4 * vz_d[l + 2] - vz_d[l + 3]);
		}
		else if (dev.xbc == closed && i == dev.nx && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))
		{
			ux_d[l] = dev.tau / dev.hx / dev.hx * (2.0 * vx_d[l] - 5.0 * vx_d[l - 1] + 4.0 * vx_d[l - 2] - vx_d[l - 3]) + dev.tau * dev.Ra / dev.Pr * (T0_d[l] + C10_d[l] + C20_d[l]) * dev.sinA;
			//uy_d[l] = dev.tau / dev.hx / dev.hx * (2 * vy_d[l] - 5 * vy_d[l - 1] + 4 * vy_d[l - 2] - vy_d[l - 3]);
			//uz_d[l] = dev.tau / dev.hx / dev.hx * (2 * vz_d[l] - 5 * vz_d[l - 1] + 4 * vz_d[l - 2] - vz_d[l - 3]);
		}

		// z
		else if (dev.zbc == closed && k == 0 && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))
		{
			//ux_d[l] = dev.tau / dev.hz / dev.hz * (2 * vx_d[l] - 5 * vx_d[l + dev.offset2] + 4 * vx_d[l + dev.offset2 * 2] - vx_d[l + dev.offset2 * 3]);
			//uy_d[l] = dev.tau / dev.hz / dev.hz * (2 * vy_d[l] - 5 * vy_d[l + dev.offset2] + 4 * vy_d[l + dev.offset2 * 2] - vy_d[l + dev.offset2 * 3]);
			uz_d[l] = dev.tau / dev.hz / dev.hz * (2.0 * vz_d[l] - 5.0 * vz_d[l + dev.offset2] + 4.0 * vz_d[l + dev.offset2 * 2] - vz_d[l + dev.offset2 * 3]);
		}
		else if (dev.zbc == closed && k == dev.nz && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))
		{
			//ux_d[l] = dev.tau / dev.hz / dev.hz * (2 * vx_d[l] - 5 * vx_d[l - dev.offset2] + 4 * vx_d[l - dev.offset2 * 2] - vx_d[l - dev.offset2 * 3]);
			//uy_d[l] = dev.tau / dev.hz / dev.hz * (2 * vy_d[l] - 5 * vy_d[l - dev.offset2] + 4 * vy_d[l - dev.offset2 * 2] - vy_d[l - dev.offset2 * 3]);
			uz_d[l] = dev.tau / dev.hz / dev.hz * (2.0 * vz_d[l] - 5.0 * vz_d[l - dev.offset2] + 4.0 * vz_d[l - dev.offset2 * 2] - vz_d[l - dev.offset2 * 3]);
		}


		/*
		PERIODIC
		*/

		// x

		else if (dev.xbc == periodic && i == 0 && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))
		{
			int ll = dev.nx - 1 + dev.offset * j + dev.offset2 * k;
			/*
			there is a slight possibility to come across the race condition effect while realizing parallel boundary condotions
			we think it is acceptabele to re-calculate some date once more to eliminate it
			ll index means the index on the opposite side but minus 1 (nx-1 here) as it goes to build a standard approach for parallel boundary conditions
			after we re-calculate the data for this index, nx-1, (that, in fact, was found above in the section for the whole inner nodes)
			*/

			ux_d[l] = vx_d[ll]
				+ dev.tau * (
					-vx_d[ll] * (vx_d[ll + 1] - vx_d[ll - 1]) / 2.0 / dev.hx					//  (ux_d*d/dx) ux_d
					- vy_d[ll] * (vx_d[ll + dev.offset] - vx_d[ll - dev.offset]) / 2.0 / dev.hy				//  (uy_d*d/dy) ux_d
					- vz_d[ll] * (vx_d[ll + dev.offset2] - vx_d[ll - dev.offset2]) / 2.0 / dev.hz				//  (uy_d*d/dy) ux_d

					+ (vx_d[ll + 1] - 2.0 * vx_d[ll] + vx_d[ll - 1]) / dev.hx / dev.hx				// (d2/dx2) ux_d
					+ (vx_d[ll + dev.offset] - 2.0 * vx_d[ll] + vx_d[ll - dev.offset]) / dev.hy / dev.hy			// (d2/dy2) ux_d
					+ (vx_d[ll + dev.offset2] - 2.0 * vx_d[ll] + vx_d[ll - dev.offset2]) / dev.hz / dev.hz			// (d2/dz2) ux_d
					);

		}
		else if (dev.xbc == periodic && i == dev.nx && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))
		{
			int ll = 1 + dev.offset * j + dev.offset2 * k;
			ux_d[l] = vx_d[ll]
				+ dev.tau * (
					-vx_d[ll] * (vx_d[ll + 1] - vx_d[ll - 1]) / 2.0 / dev.hx					//  (ux_d*d/dx) ux_d
					- vy_d[ll] * (vx_d[ll + dev.offset] - vx_d[ll - dev.offset]) / 2.0 / dev.hy				//  (uy_d*d/dy) ux_d
					- vz_d[ll] * (vx_d[ll + dev.offset2] - vx_d[ll - dev.offset2]) / 2.0 / dev.hz				//  (uy_d*d/dy) ux_d

					+ (vx_d[ll + 1] - 2.0 * vx_d[ll] + vx_d[ll - 1]) / dev.hx / dev.hx				// (d2/dx2) ux_d
					+ (vx_d[ll + dev.offset] - 2.0 * vx_d[ll] + vx_d[ll - dev.offset]) / dev.hy / dev.hy			// (d2/dy2) ux_d
					+ (vx_d[ll + dev.offset2] - 2.0 * vx_d[ll] + vx_d[ll - dev.offset2]) / dev.hz / dev.hz			// (d2/dz2) ux_d
					);
		}

		// z
		else if (dev.zbc == periodic && k == 0 && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))
		{
			int ll = i + dev.offset * j + dev.offset2 * dev.nz - dev.offset2;

			uz_d[l] = vz_d[ll]
				+ dev.tau * (
					-vx_d[ll] * (vz_d[ll + 1] - vz_d[ll - 1]) / 2.0 / dev.hx					//  (ux_d*d/dx) uz_d
					- vy_d[ll] * (vz_d[ll + dev.offset] - vz_d[ll - dev.offset]) / 2.0 / dev.hy				//  (uy_d*d/dy) uz_d
					- vz_d[ll] * (vz_d[ll + dev.offset2] - vz_d[ll - dev.offset2]) / 2.0 / dev.hz				//  (uy_d*d/dy) uz_d

					+ (vz_d[ll + 1] - 2.0 * vz_d[ll] + vz_d[ll - 1]) / dev.hx / dev.hx				// (d2/dx2) uz_d
					+ (vz_d[ll + dev.offset] - 2.0 * vz_d[ll] + vz_d[ll - dev.offset]) / dev.hy / dev.hy			// (d2/dy2) uz_d
					+ (vz_d[ll + dev.offset2] - 2.0 * vz_d[ll] + vz_d[ll - dev.offset2]) / dev.hz / dev.hz			// (d2/dz2) uz_d
					);

		}
		else if (dev.zbc == periodic && k == dev.nz && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))
		{
			int ll = i + dev.offset * j + dev.offset2;

			uz_d[l] = vz_d[ll]
				+ dev.tau * (
					-vx_d[ll] * (vz_d[ll + 1] - vz_d[ll - 1]) / 2.0 / dev.hx					//  (ux_d*d/dx) uz_d
					- vy_d[ll] * (vz_d[ll + dev.offset] - vz_d[ll - dev.offset]) / 2.0 / dev.hy				//  (uy_d*d/dy) uz_d
					- vz_d[ll] * (vz_d[ll + dev.offset2] - vz_d[ll - dev.offset2]) / 2.0 / dev.hz				//  (uy_d*d/dy) uz_d

					+ (vz_d[ll + 1] - 2.0 * vz_d[ll] + vz_d[ll - 1]) / dev.hx / dev.hx				// (d2/dx2) uz_d
					+ (vz_d[ll + dev.offset] - 2.0 * vz_d[ll] + vz_d[ll - dev.offset]) / dev.hy / dev.hy			// (d2/dy2) uz_d
					+ (vz_d[ll + dev.offset2] - 2.0 * vz_d[ll] + vz_d[ll - dev.offset2]) / dev.hz / dev.hz			// (d2/dz2) uz_d
					);

		}

		// corner points

		else if (i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N) {
			int ii = i + 1 - i / dev.nx - ceil(i / (dev.nx + 1.0));
			int jj = j + 1 - j / dev.ny - ceil(j / (dev.ny + 1.0));
			int kk = k + 1 - k / dev.nz - ceil(k / (dev.nz + 1.0));

			//ux_d[l] = 0;
			//uy_d[l] = 0;
			//uz_d[l] = 0;

			//ux_d[l] = ux_d[ii + dev.offset*jj + dev.offset2*kk];
			//uy_d[l] = uy_d[ii + dev.offset*jj + dev.offset2*kk];
			//uz_d[l] = uz_d[ii + dev.offset*jj + dev.offset2*kk];
		}


	}
}

__global__ void temperature(double* ux_d, double* uy_d, double* uz_d, double* vx_d, double* vy_d, double* vz_d, double* T_d, double* T0_d, double* C1_d, double* C10_d, double* C2_d, double* C20_d) {


	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
	unsigned int l = i + dev.offset * j + dev.offset2 * k;
	if (i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N)
	{

		/*
		INNER
		*/
		if (i > 0 && i < dev.nx && j > 0 && j < dev.ny && k > 0 && k < dev.nz)
		{

			//temperature
			T_d[l] = T0_d[l]
				+ dev.tau * (
					-vx_d[l] * (T0_d[l + 1] - T0_d[l - 1]) / (2.0 * dev.hx)					// (ux_d*d/dx) T_d
					- vy_d[l] * (T0_d[l + dev.offset] - T0_d[l - dev.offset]) / (2.0 * dev.hy)				// (uy_d*d/dy) T_d
					- vz_d[l] * (T0_d[l + dev.offset2] - T0_d[l - dev.offset2]) / (2.0 * dev.hz)				// (uz_d*d/dy) T_d

					+ ((T0_d[l + 1] - 2.0 * T0_d[l] + T0_d[l - 1]) / dev.hx / dev.hx 		// 1/Pr*(d2/dx2) T_d
						+ (T0_d[l + dev.offset] - 2.0 * T0_d[l] + T0_d[l - dev.offset]) / dev.hy / dev.hy 	// 1/Pr*(d2/dy2) T_d
						+ (T0_d[l + dev.offset2] - 2.0 * T0_d[l] + T0_d[l - dev.offset2]) / dev.hz / dev.hz) / dev.Pr	// 1/Pr*(d2/dz2) T_d
					);

		}

		/*
		UP-DOWN
		*/

		// y
		else if (j == 0 && (i > 0 && i < dev.nx && k > 0 && k < dev.nz)) 		T_d[l] = (4.0 * T0_d[l + dev.offset] - T0_d[l + dev.offset * 2]) / 3.0 - 2.0 / 3.0 * dev.hy * dev.Q;
		else if (j == dev.ny && (i > 0 && i < dev.nx && k > 0 && k < dev.nz))		T_d[l] = (4.0 * T0_d[l - dev.offset] - T0_d[l - dev.offset * 2]) / 3.0 + 2.0 / 3.0 * dev.hy * dev.Q;

		/*
		CLOSED
		*/

		// x
		else if (dev.xbc == closed && i == 0 && (j > 0 && j < dev.ny && k > 0 && k < dev.nz)) 		T_d[l] = (4.0 * T0_d[l + 1] - T0_d[l + 2]) / 3.0;
		else if (dev.xbc == closed && i == dev.nx && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		T_d[l] = (4.0 * T0_d[l - 1] - T0_d[l - 2]) / 3.0;

		// z
		else if (dev.zbc == closed && k == 0 && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		T_d[l] = (4.0 * T0_d[l + dev.offset2] - T0_d[l + dev.offset2 * 2]) / 3.0;
		else if (dev.zbc == closed && k == dev.nz && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		T_d[l] = (4.0 * T0_d[l - dev.offset2] - T0_d[l - dev.offset2 * 2]) / 3.0;


		/*
		PERIODIC
		*/

		// x

		else if (dev.xbc == periodic && i == 0 && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		T_d[l] = T0_d[dev.nx - 1 + dev.offset * j + dev.offset2 * k];
		else if (dev.xbc == periodic && i == dev.nx && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		T_d[l] = T0_d[1 + dev.offset * j + dev.offset2 * k];

		// z
		else if (dev.zbc == periodic && k == 0 && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		T_d[l] = T0_d[i + dev.offset * j + dev.offset2 * dev.nz - dev.offset2];
		else if (dev.zbc == periodic && k == dev.nz && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		T_d[l] = T0_d[i + dev.offset * j + dev.offset2];


		// corner points

		else if (i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N) {
			int ii = i + 1 - i / dev.nx - ceil(i / (dev.nx + 1.0));
			int jj = j + 1 - j / dev.ny - ceil(j / (dev.ny + 1.0));
			int kk = k + 1 - k / dev.nz - ceil(k / (dev.nz + 1.0));

			T_d[l] = T0_d[ii + dev.offset * jj + dev.offset2 * kk];
		}


	}
}
__global__ void temperature_fixed_value(double* ux_d, double* uy_d, double* uz_d, double* vx_d, double* vy_d, double* vz_d, double* T_d, double* T0_d, double* C1_d, double* C10_d, double* C2_d, double* C20_d) {


	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
	unsigned int l = i + dev.offset * j + dev.offset2 * k;
	if (i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N)
	{

		/*
		INNER
		*/
		if (i > 0 && i < dev.nx && j > 0 && j < dev.ny && k > 0 && k < dev.nz)
		{

			//temperature
			T_d[l] = T0_d[l]
				+ dev.tau * (
					-vx_d[l] * (T0_d[l + 1] - T0_d[l - 1]) / (2.0 * dev.hx)					// (ux_d*d/dx) T_d
					- vy_d[l] * (T0_d[l + dev.offset] - T0_d[l - dev.offset]) / (2.0 * dev.hy)				// (uy_d*d/dy) T_d
					- vz_d[l] * (T0_d[l + dev.offset2] - T0_d[l - dev.offset2]) / (2.0 * dev.hz)				// (uz_d*d/dy) T_d

					+ ((T0_d[l + 1] - 2.0 * T0_d[l] + T0_d[l - 1]) / dev.hx / dev.hx 		// 1/Pr*(d2/dx2) T_d
						+ (T0_d[l + dev.offset] - 2.0 * T0_d[l] + T0_d[l - dev.offset]) / dev.hy / dev.hy 	// 1/Pr*(d2/dy2) T_d
						+ (T0_d[l + dev.offset2] - 2.0 * T0_d[l] + T0_d[l - dev.offset2]) / dev.hz / dev.hz) / dev.Pr	// 1/Pr*(d2/dz2) T_d
					);

		}

		/*
		UP-DOWN
		*/

		// y
		else if (j == 0 && (i > 0 && i < dev.nx && k > 0 && k < dev.nz)) 		T_d[l] = 1;
		else if (j == dev.ny && (i > 0 && i < dev.nx && k > 0 && k < dev.nz))		T_d[l] = 0;

		/*
		CLOSED
		*/

		// x
		else if (dev.xbc == closed && i == 0 && (j > 0 && j < dev.ny && k > 0 && k < dev.nz)) 		T_d[l] = (4.0 * T0_d[l + 1] - T0_d[l + 2]) / 3.0;
		else if (dev.xbc == closed && i == dev.nx && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		T_d[l] = (4.0 * T0_d[l - 1] - T0_d[l - 2]) / 3.0;

		// z
		else if (dev.zbc == closed && k == 0 && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		T_d[l] = (4.0 * T0_d[l + dev.offset2] - T0_d[l + dev.offset2 * 2]) / 3.0;
		else if (dev.zbc == closed && k == dev.nz && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		T_d[l] = (4.0 * T0_d[l - dev.offset2] - T0_d[l - dev.offset2 * 2]) / 3.0;


		/*
		PERIODIC
		*/

		// x

		else if (dev.xbc == periodic && i == 0 && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		T_d[l] = T0_d[dev.nx - 1 + dev.offset * j + dev.offset2 * k];
		else if (dev.xbc == periodic && i == dev.nx && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		T_d[l] = T0_d[1 + dev.offset * j + dev.offset2 * k];

		// z
		else if (dev.zbc == periodic && k == 0 && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		T_d[l] = T0_d[i + dev.offset * j + dev.offset2 * dev.nz - dev.offset2];
		else if (dev.zbc == periodic && k == dev.nz && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		T_d[l] = T0_d[i + dev.offset * j + dev.offset2];


		// corner points

		else if (i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N) {
			int ii = i + 1 - i / dev.nx - ceil(i / (dev.nx + 1.0));
			int jj = j + 1 - j / dev.ny - ceil(j / (dev.ny + 1.0));
			int kk = k + 1 - k / dev.nz - ceil(k / (dev.nz + 1.0));

			T_d[l] = T0_d[ii + dev.offset * jj + dev.offset2 * kk];
		}


	}
}



__global__ void concentration1(double* ux_d, double* uy_d, double* uz_d, double* vx_d, double* vy_d, double* vz_d, double* T_d, double* T0_d, double* C1_d, double* C10_d, double* C2_d, double* C20_d) {


	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
	unsigned int l = i + dev.offset * j + dev.offset2 * k;
	if (i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N)
	{

		/*
		INNER
		*/
		if (i > 0 && i < dev.nx && j > 0 && j < dev.ny && k > 0 && k < dev.nz)
		{

			//C1
			C1_d[l] = C10_d[l]
				+ dev.tau * (
					-vx_d[l] * (C10_d[l + 1] - C10_d[l - 1]) / (2.0 * dev.hx)					// (ux_d*d/dx) C1_d
					- vy_d[l] * (C10_d[l + dev.offset] - C10_d[l - dev.offset]) / (2.0 * dev.hy)				// (uy_d*d/dy) C1_d
					- vz_d[l] * (C10_d[l + dev.offset2] - C10_d[l - dev.offset2]) / (2.0 * dev.hz)				// (uz_d*d/dy) C1_d

					+ ((C10_d[l + 1] - 2.0 * C10_d[l] + C10_d[l - 1]) / dev.hx / dev.hx 	// 1/Pr*(d2/dx2) C1_d
						+ (C10_d[l + dev.offset] - 2.0 * C10_d[l] + C10_d[l - dev.offset]) / dev.hy / dev.hy 	// 1/Pr*(d2/dy2) C1_d
						+ (C10_d[l + dev.offset2] - 2.0 * C10_d[l] + C10_d[l - dev.offset2]) / dev.hz / dev.hz) / dev.Sc11	// 1/Pr*(d2/dz2) C1_d

					- (+(T0_d[l + 1] - 2.0 * T0_d[l] + T0_d[l - 1]) / dev.hx / dev.hx 		// 1/Pr*(d2/dx2) T_d
						+ (T0_d[l + dev.offset] - 2.0 * T0_d[l] + T0_d[l - dev.offset]) / dev.hy / dev.hy 	// 1/Pr*(d2/dy2) T_d
						+ (T0_d[l + dev.offset2] - 2.0 * T0_d[l] + T0_d[l - dev.offset2]) / dev.hz / dev.hz) / dev.Sc11 * dev.psi1	// 1/Pr*(d2/dz2) T_d
					);

		}



		/*
		UP-DOWN
		*/

		// y
		else if (j == 0 && (i > 0 && i < dev.nx && k > 0 && k < dev.nz)) 		C1_d[l] = (4.0 * C10_d[l + dev.offset] - C10_d[l + dev.offset * 2]) / 3.0 - 2.0 / 3.0 * dev.hy * dev.psi1 * dev.Q;
		else if (j == dev.ny && (i > 0 && i < dev.nx && k > 0 && k < dev.nz))		C1_d[l] = (4.0 * C10_d[l - dev.offset] - C10_d[l - dev.offset * 2]) / 3.0 + 2.0 / 3.0 * dev.hy * dev.psi1 * dev.Q;

		/*
		CLOSED
		*/

		// x
		else if (dev.xbc == closed && i == 0 && (j > 0 && j < dev.ny && k > 0 && k < dev.nz)) 		C1_d[l] = (4.0 * C10_d[l + 1] - C10_d[l + 2]) / 3.0;
		else if (dev.xbc == closed && i == dev.nx && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		C1_d[l] = (4.0 * C10_d[l - 1] - C10_d[l - 2]) / 3.0;

		// z
		else if (dev.zbc == closed && k == 0 && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		C1_d[l] = (4.0 * C10_d[l + dev.offset2] - C10_d[l + dev.offset2 * 2]) / 3.0;
		else if (dev.zbc == closed && k == dev.nz && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		C1_d[l] = (4.0 * C10_d[l - dev.offset2] - C10_d[l - dev.offset2 * 2]) / 3.0;


		/*
		PERIODIC
		*/

		// x

		else if (dev.xbc == periodic && i == 0 && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		C1_d[l] = C10_d[dev.nx - 1 + dev.offset * j + dev.offset2 * k];
		else if (dev.xbc == periodic && i == dev.nx && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		C1_d[l] = C10_d[1 + dev.offset * j + dev.offset2 * k];

		// z
		else if (dev.zbc == periodic && k == 0 && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		C1_d[l] = C10_d[i + dev.offset * j + dev.offset2 * dev.nz - dev.offset2];
		else if (dev.zbc == periodic && k == dev.nz && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		C1_d[l] = C10_d[i + dev.offset * j + dev.offset2];


		// corner points

		else if (i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N) {

			int ii = i + 1 - i / dev.nx - ceil(i / (dev.nx + 1.0));
			int jj = j + 1 - j / dev.ny - ceil(j / (dev.ny + 1.0));
			int kk = k + 1 - k / dev.nz - ceil(k / (dev.nz + 1.0));

			C1_d[l] = C10_d[ii + dev.offset * jj + dev.offset2 * kk];
		}


	}
}
__global__ void concentration2(double* ux_d, double* uy_d, double* uz_d, double* vx_d, double* vy_d, double* vz_d, double* T_d, double* T0_d, double* C1_d, double* C10_d, double* C2_d, double* C20_d) {


	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
	unsigned int l = i + dev.offset * j + dev.offset2 * k;
	if (i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N)
	{

		/*
		INNER
		*/
		if (i > 0 && i < dev.nx && j > 0 && j < dev.ny && k > 0 && k < dev.nz)
		{

			//C2
			C2_d[l] = C20_d[l]
				+ dev.tau * (
					-vx_d[l] * (C20_d[l + 1] - C20_d[l - 1]) / (2.0 * dev.hx)					// (ux_d*d/dx) C2_d
					- vy_d[l] * (C20_d[l + dev.offset] - C20_d[l - dev.offset]) / (2.0 * dev.hy)				// (uy_d*d/dy) C2_d
					- vz_d[l] * (C20_d[l + dev.offset2] - C20_d[l - dev.offset2]) / (2.0 * dev.hz)				// (uz_d*d/dy) C2_d

					+ ((C20_d[l + 1] - 2.0 * C20_d[l] + C20_d[l - 1]) / dev.hx / dev.hx 	// 1/Pr*(d2/dx2) C2_d
						+ (C20_d[l + dev.offset] - 2.0 * C20_d[l] + C20_d[l - dev.offset]) / dev.hy / dev.hy 	// 1/Pr*(d2/dy2) C2_d
						+ (C20_d[l + dev.offset2] - 2.0 * C20_d[l] + C20_d[l - dev.offset2]) / dev.hz / dev.hz) / dev.Sc22	// 1/Pr*(d2/dz2) C2_d

					- (+(T0_d[l + 1] - 2.0 * T0_d[l] + T0_d[l - 1]) / dev.hx / dev.hx 		// 1/Pr*(d2/dx2) T_d
						+ (T0_d[l + dev.offset] - 2.0 * T0_d[l] + T0_d[l - dev.offset]) / dev.hy / dev.hy 	// 1/Pr*(d2/dy2) T_d
						+ (T0_d[l + dev.offset2] - 2.0 * T0_d[l] + T0_d[l - dev.offset2]) / dev.hz / dev.hz) / dev.Sc22 * dev.psi2	// 1/Pr*(d2/dz2) T_d
					);

		}



		/*
		UP-DOWN
		*/

		// y
		else if (j == 0 && (i > 0 && i < dev.nx && k > 0 && k < dev.nz)) 		C2_d[l] = (4.0 * C20_d[l + dev.offset] - C20_d[l + dev.offset * 2]) / 3.0 - 2.0 / 3.0 * dev.hy * dev.psi2 * dev.Q;
		else if (j == dev.ny && (i > 0 && i < dev.nx && k > 0 && k < dev.nz))		C2_d[l] = (4.0 * C20_d[l - dev.offset] - C20_d[l - dev.offset * 2]) / 3.0 + 2.0 / 3.0 * dev.hy * dev.psi2 * dev.Q;

		/*
		CLOSED
		*/

		// x
		else if (dev.xbc == closed && i == 0 && (j > 0 && j < dev.ny && k > 0 && k < dev.nz)) 		C2_d[l] = (4.0 * C20_d[l + 1] - C20_d[l + 2]) / 3.0;
		else if (dev.xbc == closed && i == dev.nx && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		C2_d[l] = (4.0 * C20_d[l - 1] - C20_d[l - 2]) / 3.0;

		// z
		else if (dev.zbc == closed && k == 0 && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		C2_d[l] = (4.0 * C20_d[l + dev.offset2] - C20_d[l + dev.offset2 * 2]) / 3.0;
		else if (dev.zbc == closed && k == dev.nz && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		C2_d[l] = (4.0 * C20_d[l - dev.offset2] - C20_d[l - dev.offset2 * 2]) / 3.0;


		/*
		PERIODIC
		*/

		// x

		else if (dev.xbc == periodic && i == 0 && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		C2_d[l] = C20_d[dev.nx - 1 + dev.offset * j + dev.offset2 * k];
		else if (dev.xbc == periodic && i == dev.nx && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		C2_d[l] = C20_d[1 + dev.offset * j + dev.offset2 * k];

		// z
		else if (dev.zbc == periodic && k == 0 && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		C2_d[l] = C20_d[i + dev.offset * j + dev.offset2 * dev.nz - dev.offset2];
		else if (dev.zbc == periodic && k == dev.nz && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		C2_d[l] = C20_d[i + dev.offset * j + dev.offset2];


		// corner points

		else if (i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N) {

			int ii = i + 1 - i / dev.nx - ceil(i / (dev.nx + 1.0));
			int jj = j + 1 - j / dev.ny - ceil(j / (dev.ny + 1.0));
			int kk = k + 1 - k / dev.nz - ceil(k / (dev.nz + 1.0));

			C2_d[l] = C20_d[ii + dev.offset * jj + dev.offset2 * kk];
		}


	}
}



__global__ void velocity_correction(double* ux_d, double* uy_d, double* uz_d, double* vx_d, double* vy_d, double* vz_d, double* p_d) {


	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
	unsigned int l = i + dev.offset * j + dev.offset2 * k;

	if (dev.xbc == closed && i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N)
	{
		if (i > 0 && i < dev.nx && j > 0 && j < dev.ny && k > 0 && k < dev.nz)
		{
			vx_d[l] = ux_d[l] - dev.tau / (2.0 * dev.hx) * (p_d[l + 1] - p_d[l - 1]);
			vy_d[l] = uy_d[l] - dev.tau / (2.0 * dev.hy) * (p_d[l + dev.offset] - p_d[l - dev.offset]);
			vz_d[l] = uz_d[l] - dev.tau / (2.0 * dev.hz) * (p_d[l + dev.offset2] - p_d[l - dev.offset2]);
		}
		else if (i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N) {
			vx_d[l] = 0; vy_d[l] = 0; 	vz_d[l] = 0;
		}

	}


	else if (dev.xbc == periodic && i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N)
	{
		if (i > 0 && i < dev.nx && j > 0 && j < dev.ny && k > 0 && k < dev.nz)
		{
			vx_d[l] = ux_d[l] - dev.tau / 2.0 / dev.hx * (p_d[l + 1] - p_d[l - 1]);
			vy_d[l] = uy_d[l] - dev.tau / 2.0 / dev.hy * (p_d[l + dev.offset] - p_d[l - dev.offset]);
			vz_d[l] = uz_d[l] - dev.tau / 2.0 / dev.hz * (p_d[l + dev.offset2] - p_d[l - dev.offset2]);
		}
		else if (j == 0 && (i <= dev.nx && k <= dev.nz))
		{
			vx_d[l] = 0.0; vy_d[l] = 0.0; vz_d[l] = 0.0;
		}
		else if (j == dev.ny && (i <= dev.nx && k <= dev.nz))
		{
			vx_d[l] = 0.0; vy_d[l] = 0.0; vz_d[l] = 0.0;
		}
		else if (i == 0 && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))
		{
			int ll = dev.nx - 1 + dev.offset * j + dev.offset2 * k;
			vx_d[l] = ux_d[ll] - dev.tau / 2.0 / dev.hx * (p_d[ll + 1] - p_d[ll - 1]);
			vy_d[l] = uy_d[ll] - dev.tau / 2.0 / dev.hy * (p_d[ll + dev.offset] - p_d[ll - dev.offset]);
			vz_d[l] = uz_d[ll] - dev.tau / 2.0 / dev.hz * (p_d[ll + dev.offset2] - p_d[ll - dev.offset2]);
		}
		else if (i == dev.nx && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))
		{
			int ll = 1 + dev.offset * j + dev.offset2 * k;
			vx_d[l] = ux_d[ll] - dev.tau / 2.0 / dev.hx * (p_d[ll + 1] - p_d[ll - 1]);
			vy_d[l] = uy_d[ll] - dev.tau / 2.0 / dev.hy * (p_d[ll + dev.offset] - p_d[ll - dev.offset]);
			vz_d[l] = uz_d[ll] - dev.tau / 2.0 / dev.hz * (p_d[ll + dev.offset2] - p_d[ll - dev.offset2]);
		}
		else if (k == 0 && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))
		{
			int ll = i + dev.offset * j + dev.offset2 * dev.nz - dev.offset2;
			vx_d[l] = ux_d[ll] - dev.tau / 2.0 / dev.hx * (p_d[ll + 1] - p_d[ll - 1]);
			vy_d[l] = uy_d[ll] - dev.tau / 2.0 / dev.hy * (p_d[ll + dev.offset] - p_d[ll - dev.offset]);
			vz_d[l] = uz_d[ll] - dev.tau / 2.0 / dev.hz * (p_d[ll + dev.offset2] - p_d[ll - dev.offset2]);
		}
		else if (k == dev.nz && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))
		{
			int ll = i + dev.offset * j + dev.offset2;
			vx_d[l] = ux_d[ll] - dev.tau / 2.0 / dev.hx * (p_d[ll + 1] - p_d[ll - 1]);
			vy_d[l] = uy_d[ll] - dev.tau / 2.0 / dev.hy * (p_d[ll + dev.offset] - p_d[ll - dev.offset]);
			vz_d[l] = uz_d[ll] - dev.tau / 2.0 / dev.hz * (p_d[ll + dev.offset2] - p_d[ll - dev.offset2]);
		}
	}

	else if (i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N) {
		int ii = i + 1 - i / dev.nx - ceil(i / (dev.nx + 1.0));
		int jj = j + 1 - j / dev.ny - ceil(j / (dev.ny + 1.0));
		int kk = k + 1 - k / dev.nz - ceil(k / (dev.nz + 1.0));

		vx_d[l] = 0;
		vy_d[l] = 0;
		vz_d[l] = 0;

	}

}


__global__ void Poisson(double* ux_d, double* uy_d, double* uz_d, double* p_d, double* p0_d)
{
	//период условие отключено
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
	unsigned int l = i + dev.offset * j + dev.offset2 * k;
	//double psiav0 = 0.0; double psiav = 0.0; double eps = 1.0; int k = 0;

	/*

	closed

	*/

	if (dev.xbc == closed && i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N)
	{
		if (i > 0 && i < dev.nx && j > 0 && j < dev.ny && k > 0 && k < dev.nz)
		{
			p_d[l] =
				-(ux_d[l + 1] - ux_d[l - 1]  /*/ dev.tau / 2.0 / dev.hx */
					+ uy_d[l + dev.offset] - uy_d[l - dev.offset]  /* / dev.tau / 2.0 / dev.hy */
					+ uz_d[l + dev.offset2] - uz_d[l - dev.offset2]) / dev.tau / 2.0 / dev.hz;

			if (i == 1) 				p_d[l] += 2.0 / 3.0 / dev.hx / dev.hx * (p0_d[l + 1] - p0_d[l] - dev.hx / dev.tau * ux_d[l - 1]);
			else if (i == dev.nx - 1) 	p_d[l] += 2.0 / 3.0 / dev.hx / dev.hx * (p0_d[l - 1] - p0_d[l] + dev.hx / dev.tau * ux_d[l + 1]);
			else						p_d[l] += 1.0 / dev.hx / dev.hx * (p0_d[l + 1] + p0_d[l - 1] - 2.0 * p0_d[l]);

			if (j == 1)					p_d[l] += 2.0 / 3.0 / dev.hy / dev.hy * (p0_d[l + dev.offset] - p0_d[l] - dev.hy / dev.tau * uy_d[l - dev.offset]);
			else if (j == dev.ny - 1) 	p_d[l] += 2.0 / 3.0 / dev.hy / dev.hy * (p0_d[l - dev.offset] - p0_d[l] + dev.hy / dev.tau * uy_d[l + dev.offset]);
			else						p_d[l] += 1.0 / dev.hy / dev.hy * (p0_d[l + dev.offset] + p0_d[l - dev.offset] - 2.0 * p0_d[l]);

			if (k == 1)					p_d[l] += 2.0 / 3.0 / dev.hz / dev.hz * (p0_d[l + dev.offset2] - p0_d[l] - dev.hz / dev.tau * uz_d[l - dev.offset2]);
			else if (k == dev.nz - 1)		p_d[l] += 2.0 / 3.0 / dev.hz / dev.hz * (p0_d[l - dev.offset2] - p0_d[l] + dev.hz / dev.tau * uz_d[l + dev.offset2]);
			else						p_d[l] += 1.0 / dev.hz / dev.hz * (p0_d[l + dev.offset2] + p0_d[l - dev.offset2] - 2.0 * p0_d[l]);

			p_d[l] *= dev.tau_p;
			p_d[l] += p0_d[l];
		}


		else if (j == 0 && (i > 0 && i < dev.nx && k > 0 && k < dev.nz))		p_d[l] = (4.0 * p0_d[l + dev.offset] - p0_d[l + dev.offset * 2]) / 3.0 - uy_d[l] * 2.0 * dev.hy / dev.tau / 3.0;
		else if (j == dev.ny && (i > 0 && i < dev.nx && k > 0 && k < dev.nz))		p_d[l] = (4.0 * p0_d[l - dev.offset] - p0_d[l - dev.offset * 2]) / 3.0 + uy_d[l] * 2.0 * dev.hy / dev.tau / 3.0;
		else if (i == 0 && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		p_d[l] = (4.0 * p0_d[l + 1] - p0_d[l + 2]) / 3.0 - ux_d[l] * 2.0 * dev.hx / dev.tau / 3.0;
		else if (i == dev.nx && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		p_d[l] = (4.0 * p0_d[l - 1] - p0_d[l - 2]) / 3.0 + ux_d[l] * 2.0 * dev.hx / dev.tau / 3.0;
		else if (k == 0 && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		p_d[l] = (4.0 * p0_d[l + dev.offset2] - p0_d[l + dev.offset2 * 2]) / 3.0 - uz_d[l] * 2.0 * dev.hz / dev.tau / 3.0;
		else if (k == dev.nz && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		p_d[l] = (4.0 * p0_d[l - dev.offset2] - p0_d[l - dev.offset2 * 2]) / 3.0 + uz_d[l] * 2.0 * dev.hz / dev.tau / 3.0;

		else if (i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N) {
			int ii = i + 1 - i / dev.nx - ceil(i / (dev.nx + 1.0));
			int jj = j + 1 - j / dev.ny - ceil(j / (dev.ny + 1.0));
			int kk = k + 1 - k / dev.nz - ceil(k / (dev.nz + 1.0));
			p_d[l] = p0_d[ii + dev.offset * jj + dev.offset2 * kk];
			//p_d[l] = 0;
			//int ii = 0;
			//int jj = 0;
			//int kk = 0;

			//if (i == 0)	ii = 1;		if (i == dev.nx) ii = -1;
			//if (j == 0)	jj = 1;		if (j == dev.ny) jj = -1;
			//if (k == 0)	kk = 1;		if (k == dev.nz) kk = -1;

			//p_d[l] = (4.0*p0_d[l + 1 * ii + dev.offset*jj + dev.offset2*kk] - p0_d[l + 2 * ii + dev.offset*jj + dev.offset2*kk]) / 3.0
			//	- ii* ux_d[l] * 2.0 / 3.0*dev.hx / dev.tau
			//	- jj* uy_d[l] * 2.0 / 3.0*dev.hy / dev.tau
			//	- kk* uz_d[l] * 2.0 / 3.0*dev.hz / dev.tau;
		}
	}

	/*

	periodic

	*/


	else if (dev.xbc == periodic && i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N)
	{
		if (i > 0 && i < dev.nx && j > 0 && j < dev.ny && k > 0 && k < dev.nz)
		{
			p_d[l] =
				-(ux_d[l + 1] - ux_d[l - 1]  /*/ dev.tau / 2.0 / dev.hx */
					+ uy_d[l + dev.offset] - uy_d[l - dev.offset]  /* / dev.tau / 2.0 / dev.hy */
					+ uz_d[l + dev.offset2] - uz_d[l - dev.offset2]) / dev.tau / 2.0 / dev.hz;

			/*if (i == 1) 				p_d[l] += 2.0 / 3.0 / dev.hx / dev.hx*(p0_d[l + 1] - p0_d[l] - dev.hx / dev.tau*ux_d[l - 1]);
			else if (i == dev.nx - 1) 	p_d[l] += 2.0 / 3.0 / dev.hx / dev.hx*(p0_d[l - 1] - p0_d[l] + dev.hx / dev.tau*ux_d[l + 1]);
			else*/						p_d[l] += 1.0 / dev.hx / dev.hx * (p0_d[l + 1] + p0_d[l - 1] - 2.0 * p0_d[l]);

			if (j == 1)					p_d[l] += 2.0 / 3.0 / dev.hy / dev.hy * (p0_d[l + dev.offset] - p0_d[l] - dev.hy / dev.tau * uy_d[l - dev.offset]);
			else if (j == dev.ny - 1) 	p_d[l] += 2.0 / 3.0 / dev.hy / dev.hy * (p0_d[l - dev.offset] - p0_d[l] + dev.hy / dev.tau * uy_d[l + dev.offset]);
			else						p_d[l] += 1.0 / dev.hy / dev.hy * (p0_d[l + dev.offset] + p0_d[l - dev.offset] - 2.0 * p0_d[l]);

			/*if (k == 1)					p_d[l] += 2.0 / 3.0 / dev.hz / dev.hz*(p0_d[l + dev.offset2] - p0_d[l] - dev.hz / dev.tau*uz_d[l - dev.offset2]);
			//else if (k == dev.nz - 1)		p_d[l] += 2.0 / 3.0 / dev.hz / dev.hz*(p0_d[l - dev.offset2] - p0_d[l] + dev.hz / dev.tau*uz_d[l + dev.offset2]);
			*/							p_d[l] += 1.0 / dev.hz / dev.hz * (p0_d[l + dev.offset2] + p0_d[l - dev.offset2] - 2.0 * p0_d[l]);

			p_d[l] *= dev.tau_p;
			p_d[l] += p0_d[l];
		}

		else if (j == 0 && (i > 0 && i < dev.nx && k <= dev.nz))		p_d[l] = (4.0 * p0_d[l + dev.offset] - p0_d[l + dev.offset * 2]) / 3.0 - uy_d[l] * 2.0 / 3.0 * dev.hy / dev.tau;
		else if (j == dev.ny && (i > 0 && i < dev.nx && k <= dev.nz))	p_d[l] = (4.0 * p0_d[l - dev.offset] - p0_d[l - dev.offset * 2]) / 3.0 + uy_d[l] * 2.0 / 3.0 * dev.hy / dev.tau;
		else if (i == 0 && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		p_d[l] = p0_d[dev.nx - 1 + dev.offset * j + dev.offset2 * k];
		else if (i == dev.nx && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		p_d[l] = p0_d[1 + dev.offset * j + dev.offset2 * k];
		else if (k == 0 && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		p_d[l] = p0_d[i + dev.offset * j + dev.offset2 * dev.nz - dev.offset2];
		else if (k == dev.nz && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		p_d[l] = p0_d[i + dev.offset * j + dev.offset2];

		else if (i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N) {

			int ii = i + 1 - i / dev.nx - ceil(i / (dev.nx + 1.0));
			int jj = j + 1 - j / dev.ny - ceil(j / (dev.ny + 1.0));
			int kk = k + 1 - k / dev.nz - ceil(k / (dev.nz + 1.0));
			p_d[l] = p0_d[ii + dev.offset * jj + dev.offset2 * kk];
			//int ii = 0;
			//int jj = 0;
			//int kk = 0;

			//if (i == 0)	ii = 1;		if (i == dev.nx) ii = -1;
			//if (j == 0)	jj = 1;		if (j == dev.ny) jj = -1;
			//if (k == 0)	kk = 1;		if (k == dev.nz) kk = -1;

			//p_d[l] = (4.0*p0_d[l + 1 * ii + dev.offset*jj + dev.offset2*kk] - p0_d[l + 2 * ii + dev.offset*jj + dev.offset2*kk]) / 3.0
			//	- ii* ux_d[l] * 2.0 / 3.0*dev.hx / dev.tau
			//	- jj* uy_d[l] * 2.0 / 3.0*dev.hy / dev.tau
			//	- kk* uz_d[l] * 2.0 / 3.0*dev.hz / dev.tau;
		}
	}

}

__global__ void Poisson_v2(double* ux_d, double* uy_d, double* uz_d, double* p_d, double* p0_d, int order)
{
	//период условие отключено
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
	unsigned int l = i + dev.offset * j + dev.offset2 * k;
	//double psiav0 = 0.0; double psiav = 0.0; double eps = 1.0; int k = 0;

	if ((i + j) % 2 == order) return;

	/*

	closed

	*/

	if (dev.xbc == closed && i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N)
	{
		if (i > 0 && i < dev.nx && j > 0 && j < dev.ny && k > 0 && k < dev.nz)
		{
			p_d[l] =
				-(ux_d[l + 1] - ux_d[l - 1]  /*/ dev.tau / 2.0 / dev.hx */
					+ uy_d[l + dev.offset] - uy_d[l - dev.offset]  /* / dev.tau / 2.0 / dev.hy */
					+ uz_d[l + dev.offset2] - uz_d[l - dev.offset2]) / dev.tau / 2.0 / dev.hz;

			if (i == 1) 				p_d[l] += 2.0 / 3.0 / dev.hx / dev.hx * (p0_d[l + 1] - p0_d[l] - dev.hx / dev.tau * ux_d[l - 1]);
			else if (i == dev.nx - 1) 	p_d[l] += 2.0 / 3.0 / dev.hx / dev.hx * (p0_d[l - 1] - p0_d[l] + dev.hx / dev.tau * ux_d[l + 1]);
			else						p_d[l] += 1.0 / dev.hx / dev.hx * (p0_d[l + 1] + p0_d[l - 1] - 2.0 * p0_d[l]);

			if (j == 1)					p_d[l] += 2.0 / 3.0 / dev.hy / dev.hy * (p0_d[l + dev.offset] - p0_d[l] - dev.hy / dev.tau * uy_d[l - dev.offset]);
			else if (j == dev.ny - 1) 	p_d[l] += 2.0 / 3.0 / dev.hy / dev.hy * (p0_d[l - dev.offset] - p0_d[l] + dev.hy / dev.tau * uy_d[l + dev.offset]);
			else						p_d[l] += 1.0 / dev.hy / dev.hy * (p0_d[l + dev.offset] + p0_d[l - dev.offset] - 2.0 * p0_d[l]);

			if (k == 1)					p_d[l] += 2.0 / 3.0 / dev.hz / dev.hz * (p0_d[l + dev.offset2] - p0_d[l] - dev.hz / dev.tau * uz_d[l - dev.offset2]);
			else if (k == dev.nz - 1)		p_d[l] += 2.0 / 3.0 / dev.hz / dev.hz * (p0_d[l - dev.offset2] - p0_d[l] + dev.hz / dev.tau * uz_d[l + dev.offset2]);
			else						p_d[l] += 1.0 / dev.hz / dev.hz * (p0_d[l + dev.offset2] + p0_d[l - dev.offset2] - 2.0 * p0_d[l]);

			p_d[l] *= dev.tau_p;
			p_d[l] += p0_d[l];
		}


		else if (j == 0 && (i > 0 && i < dev.nx && k > 0 && k < dev.nz))		p_d[l] = (4.0 * p0_d[l + dev.offset] - p0_d[l + dev.offset * 2]) / 3.0 - uy_d[l] * 2.0 * dev.hy / dev.tau / 3.0;
		else if (j == dev.ny && (i > 0 && i < dev.nx && k > 0 && k < dev.nz))		p_d[l] = (4.0 * p0_d[l - dev.offset] - p0_d[l - dev.offset * 2]) / 3.0 + uy_d[l] * 2.0 * dev.hy / dev.tau / 3.0;
		else if (i == 0 && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		p_d[l] = (4.0 * p0_d[l + 1] - p0_d[l + 2]) / 3.0 - ux_d[l] * 2.0 * dev.hx / dev.tau / 3.0;
		else if (i == dev.nx && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		p_d[l] = (4.0 * p0_d[l - 1] - p0_d[l - 2]) / 3.0 + ux_d[l] * 2.0 * dev.hx / dev.tau / 3.0;
		else if (k == 0 && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		p_d[l] = (4.0 * p0_d[l + dev.offset2] - p0_d[l + dev.offset2 * 2]) / 3.0 - uz_d[l] * 2.0 * dev.hz / dev.tau / 3.0;
		else if (k == dev.nz && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		p_d[l] = (4.0 * p0_d[l - dev.offset2] - p0_d[l - dev.offset2 * 2]) / 3.0 + uz_d[l] * 2.0 * dev.hz / dev.tau / 3.0;

		else if (i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N) {
			int ii = i + 1 - i / dev.nx - ceil(i / (dev.nx + 1.0));
			int jj = j + 1 - j / dev.ny - ceil(j / (dev.ny + 1.0));
			int kk = k + 1 - k / dev.nz - ceil(k / (dev.nz + 1.0));
			p_d[l] = p0_d[ii + dev.offset * jj + dev.offset2 * kk];
		}
	}

	/*

	periodic

	*/


	else if (dev.xbc == periodic && i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N)
	{
		if (i > 0 && i < dev.nx && j > 0 && j < dev.ny && k > 0 && k < dev.nz)
		{
			p_d[l] =
				-(ux_d[l + 1] - ux_d[l - 1]  /*/ dev.tau / 2.0 / dev.hx */
					+ uy_d[l + dev.offset] - uy_d[l - dev.offset]  /* / dev.tau / 2.0 / dev.hy */
					+ uz_d[l + dev.offset2] - uz_d[l - dev.offset2]) / dev.tau / 2.0 / dev.hz;

			/*if (i == 1) 				p_d[l] += 2.0 / 3.0 / dev.hx / dev.hx*(p0_d[l + 1] - p0_d[l] - dev.hx / dev.tau*ux_d[l - 1]);
			else if (i == dev.nx - 1) 	p_d[l] += 2.0 / 3.0 / dev.hx / dev.hx*(p0_d[l - 1] - p0_d[l] + dev.hx / dev.tau*ux_d[l + 1]);
			else*/						p_d[l] += 1.0 / dev.hx / dev.hx * (p0_d[l + 1] + p0_d[l - 1] - 2.0 * p0_d[l]);

			if (j == 1)					p_d[l] += 2.0 / 3.0 / dev.hy / dev.hy * (p0_d[l + dev.offset] - p0_d[l] - dev.hy / dev.tau * uy_d[l - dev.offset]);
			else if (j == dev.ny - 1) 	p_d[l] += 2.0 / 3.0 / dev.hy / dev.hy * (p0_d[l - dev.offset] - p0_d[l] + dev.hy / dev.tau * uy_d[l + dev.offset]);
			else						p_d[l] += 1.0 / dev.hy / dev.hy * (p0_d[l + dev.offset] + p0_d[l - dev.offset] - 2.0 * p0_d[l]);

			/*if (k == 1)					p_d[l] += 2.0 / 3.0 / dev.hz / dev.hz*(p0_d[l + dev.offset2] - p0_d[l] - dev.hz / dev.tau*uz_d[l - dev.offset2]);
			//else if (k == dev.nz - 1)		p_d[l] += 2.0 / 3.0 / dev.hz / dev.hz*(p0_d[l - dev.offset2] - p0_d[l] + dev.hz / dev.tau*uz_d[l + dev.offset2]);
			*/							p_d[l] += 1.0 / dev.hz / dev.hz * (p0_d[l + dev.offset2] + p0_d[l - dev.offset2] - 2.0 * p0_d[l]);

			p_d[l] *= dev.tau_p;
			p_d[l] += p0_d[l];
		}

		else if (j == 0 && (i > 0 && i < dev.nx && k <= dev.nz))		p_d[l] = (4.0 * p0_d[l + dev.offset] - p0_d[l + dev.offset * 2]) / 3.0 - uy_d[l] * 2.0 / 3.0 * dev.hy / dev.tau;
		else if (j == dev.ny && (i > 0 && i < dev.nx && k <= dev.nz))	p_d[l] = (4.0 * p0_d[l - dev.offset] - p0_d[l - dev.offset * 2]) / 3.0 + uy_d[l] * 2.0 / 3.0 * dev.hy / dev.tau;
		else if (i == 0 && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		p_d[l] = p0_d[dev.nx - 1 + dev.offset * j + dev.offset2 * k];
		else if (i == dev.nx && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))		p_d[l] = p0_d[1 + dev.offset * j + dev.offset2 * k];
		else if (k == 0 && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		p_d[l] = p0_d[i + dev.offset * j + dev.offset2 * dev.nz - dev.offset2];
		else if (k == dev.nz && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))		p_d[l] = p0_d[i + dev.offset * j + dev.offset2];

		else if (i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N) {

			int ii = i + 1 - i / dev.nx - ceil(i / (dev.nx + 1.0));
			int jj = j + 1 - j / dev.ny - ceil(j / (dev.ny + 1.0));
			int kk = k + 1 - k / dev.nz - ceil(k / (dev.nz + 1.0));
			p_d[l] = p0_d[ii + dev.offset * jj + dev.offset2 * kk];
		}
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

