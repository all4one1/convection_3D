#pragma once

#include "../common/types.h"


inline void Nu_y_for_fixed_flux(const Configuration& c,	const double* f, double& Nu)
{
	double s_top = 0;
	double s_bottom = 0;

	for (unsigned int k = 1; k < c.nz; ++k)
	{
		for (unsigned int i = 1; i < c.nx; ++i)
		{
			const unsigned int top =
				i + c.offset * c.ny + c.offset2 * k;
			const unsigned int bottom =
				i + c.offset2 * k;

			s_top += f[top];
			s_bottom += f[bottom];
		}
	}

	const unsigned int n = (c.nx - 1) * (c.nz - 1);
	s_top /= n;
	s_bottom /= n;

	Nu = 1.0 / (s_bottom - s_top);
}            

inline void Nu_y_for_fixed_temperature(const Configuration& c, double* f,
	double& Nu_top, double& Nu_down, double& Nu, double plus_const = 0)
{
	Nu_top = 0;
	Nu_down = 0;

	auto dy1 = [&](unsigned int l)
		{
			//return abs((f[l + c.offset] - f[l])) / c.hy;
			return abs(-3 * f[l] + 4 * f[l + c.offset] - f[l + 2 * c.offset]) / (2 * c.hy);
		};
	auto dy1_ = [&](unsigned int l)
		{
			//return abs((f[l] - f[l - c.offset])) / c.hy;
			return abs(3 * f[l] - 4 * f[l - c.offset] + f[l - 2 * c.offset]) / (2 * c.hy);
		};

	unsigned int l;
	for (unsigned int i = 1; i <= c.nx - 2; i++)
	{
		for (unsigned int k = 1; k <= c.nz - 2; k++)
		{

			l = i + c.offset2 * k;
			Nu_down += (dy1(l) + dy1(l + 1) + dy1(l + c.offset2) + dy1(l + 1 + c.offset2)) / 4.0 * c.hx * c.hz;
			l = i + c.offset * c.ny + c.offset2 * k;
			Nu_top += (dy1_(l) + dy1_(l + 1) + dy1_(l + c.offset2) + dy1_(l + 1 + c.offset2)) / 4.0 * c.hx * c.hz;
		}
	}

	double S = (c.Lx - 2 * c.hx) * (c.Lz - 2 * c.hz);
	Nu_down = Nu_down / (S)+plus_const;
	Nu_top = Nu_top / (S)+plus_const;

	Nu = 0.5 * (Nu_top + Nu_down);
}


double max_vertical_difference(const Configuration& c, double* f) {
	double max = 0, dif = 0;

	for (int i = 0; i <= c.nx; i++) {
		for (int k = 0; k <= c.nz; k++) {
			dif = f[i + c.offset2 * k] - f[i + c.offset * c.ny + c.offset2 * k];
			dif = abs(dif);
			if (dif > max) max = dif;
		}
	}

	return max;
}

void velocity(const Configuration& c, double vx[], double vy[], double vz[], double& Ek, double& Vmax) {
	double V2 = 0;
	Ek = 0.0; Vmax = 0.0;

	for (unsigned int k = 1; k <= c.nz - 1; k++) {
		for (unsigned int j = 1; j <= c.ny - 1; j++) {
			for (unsigned int i = 1; i <= c.nx - 1; i++) {
				unsigned int l = i + j * c.offset + k * c.offset2;
				V2 = + vx[l] * vx[l]	+ vy[l] * vy[l]	+ vz[l] * vz[l];
				if (sqrt(V2) > Vmax) Vmax = sqrt(V2);
				
				Ek += V2;
			}
		}
	}

	Ek = Ek / 2.0 * c.dV;
}

void angular_momentum(const Configuration& c, double* vx, double* vy, double* vz,	double& AMx, double& AMy, double& AMz, double& AMabs)
{
	AMx = 0; AMy = 0; AMz = 0; AMabs = 0;
	double x0 = c.Lx * 0.5;
	double y0 = c.Ly * 0.5;
	double z0 = c.Lz * 0.5;
	double x, y, z;
	unsigned int l;

	for (unsigned int k = 1; k <= c.nz - 1; k++) {
		for (unsigned int j = 1; j <= c.ny - 1; j++) {
			for (unsigned int i = 1; i <= c.nx - 1; i++) {
				l = i + c.offset * j + c.offset2 * k;
				x = i * c.hx; 
				y = j * c.hy; 
				z = k * c.hz;
				AMx += (z - z0) * vy[l] - (y - y0) * vz[l];
				AMy += (x - x0) * vz[l] - (z - z0) * vx[l];
				AMz += (y - y0) * vx[l] - (x - x0) * vy[l];
			}
		}
	}

	AMx *= c.dV;
	AMy *= c.dV;
	AMz *= c.dV;

	AMabs = sqrt(AMx * AMx + AMy * AMy + AMz * AMz);
}


void maxval_index(const Configuration& c, double* f,
	unsigned int& mx, unsigned int& my, unsigned int& mz, double& max)
{
	max = f[0];
	mx = 0; my = 0; mz = 0;


	for (unsigned int i = 0; i <= c.nx; i++) {
		for (unsigned int j = 0; j <= c.ny; j++) {
			for (unsigned int k = 0; k <= c.nz; k++) {
				if (abs(f[i + c.offset * j + c.offset2 * k]) > abs(max))
				{
					max = (f[i + c.offset * j + c.offset2 * k]);
					mx = i;
					my = j;
					mz = k;
				}
			}
		}
	}

}

void find_max_abs_xz_section_with_index(const Configuration& c, double* f,	unsigned int& mx, unsigned int& mz, double& max)
{
	max = f[0];
	mx = 0; mz = 0;

	unsigned int j = c.ny / 2;

	for (unsigned int i = 1; i <= c.nx - 1; i++) {
		for (unsigned int k = 1; k <= c.nz - 1; k++) {
			unsigned int l = i + c.offset * j + c.offset2 * k;
			if (abs(f[l]) > abs(max))
			{
				max = (f[l]);
				mx = i;
				mz = k;
			}
		}
	}

}

void find_max_Vabs_xz_section_with_index(const Configuration& c, double* vx, double* vy, double* vz,	unsigned int& mx, unsigned int& mz, double& max)
{
	max = sqrt(vx[0] * vx[0] + vy[0] * vy[0] + vz[0] * vz[0]);
	mx = 0; mz = 0;

	unsigned int j = c.ny / 2;
	double V = 0;
	for (unsigned int i = 1; i <= c.nx - 1; i++) {
		for (unsigned int k = 1; k <= c.nz - 1; k++) {
			unsigned int l = i + c.offset * j + c.offset2 * k;
			V = sqrt(vx[l] * vx[l] + vy[l] * vy[l] + vz[l] * vz[l]);
			if (V > max)
			{
				max = V;
				mx = i;
				mz = k;
			}
		}
	}

}


