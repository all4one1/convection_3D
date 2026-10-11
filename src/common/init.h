#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

//globals
std::string folder_fields;
std::string folder_final;
std::string folder_lines;


void init_parameters(Configuration& c, RUN_STATE& state)
{
#ifdef _WIN32
	CreateDirectoryA("fields", NULL);
	CreateDirectoryA("fields\\section", NULL);
	CreateDirectoryA("fields\\section\\main", NULL);
	CreateDirectoryA("fields\\section\\cross", NULL);
	CreateDirectoryA("fields\\section\\face", NULL);
	CreateDirectoryA("fields\\section2", NULL);
	CreateDirectoryA("fields\\section2\\main", NULL);
	CreateDirectoryA("fields\\section2\\cross", NULL);
	CreateDirectoryA("fields\\section2\\face", NULL);
	CreateDirectoryA("fields\\full", NULL);
	CreateDirectoryA("fields\\final", NULL);
#endif

	//addFolder("fields", folder_fields);
	//addFolder("final", folder_final);
	//addFolder("lines", folder_lines);

	const ReadingFile par("parameters.txt");
	state.read_recovery = par.required<int>("read_recovery");
	state.timeq_min = par.required<double>("timeq_min");
	state.timeq_max = par.required<double>("timeq_max");
	state.freq_fields = par.required<double>("freq_fields");
	state.call_max = par.required<int>("call_max");

	c.Lx = par.required<double>("Lx");
	c.Ly = par.required<double>("Ly");
	c.Lz = par.required<double>("Lz");
	c.nx = par.required<unsigned int>("nx");
	c.ny = par.required<unsigned int>("ny");
	c.nz = par.required<unsigned int>("nz");
	c.Ra = par.required<double>("Ra");
	c.incr_parameter = par.required<double>("increment");
	c.Pr = par.required<double>("Pr");
	c.Sc11 = par.required<double>("Sc11");
	c.Sc12 = par.required<double>("Sc12");
	c.Sc21 = par.required<double>("Sc21");
	c.Sc22 = par.required<double>("Sc22");
	c.psi1 = par.required<double>("psi1");
	c.psiS = par.required<double>("psiS");
	c.Q = par.required<double>("Q");
	c.tau = par.required<double>("tau");
	c.alpha = par.required<double>("alpha");

	auto read_boundary = [](const ReadingFile& parameters, const std::string& name)
		{
			const int value = parameters.required<int>(name);
			return static_cast<bc_type>(value);
		};

	c.xbc = read_boundary(par, "xbc");
	c.ybc = read_boundary(par, "ybc");
	c.zbc = read_boundary(par, "zbc");



	c.dim = 3;
	c.disc = FD;
	c.hx = c.Lx / static_cast<double>(c.nx);
	c.hy = c.Ly / static_cast<double>(c.ny);
	c.hz = c.Lz / static_cast<double>(c.nz);
	c.Sx = c.hy * c.hz;
	c.Sy = c.hx * c.hz;
	c.Sz = c.hx * c.hy;
	c.dV = c.hx * c.hy * c.hz;

	const std::size_t size_x = static_cast<std::size_t>(c.nx) + 1;
	const std::size_t size_y = static_cast<std::size_t>(c.ny) + 1;
	const std::size_t size_z = static_cast<std::size_t>(c.nz) + 1;
	const std::size_t plane_size = size_x * size_y;

	const std::size_t count = plane_size * size_z;
	c.offset = static_cast<unsigned int>(size_x);
	c.offset2 = static_cast<unsigned int>(plane_size);
	c.N = static_cast<unsigned int>(count);
	c.Nbytes = count * sizeof(double);
	c.tau_p = 0.1 * c.hx * c.hx;
	c.psi2 = c.psiS - c.psi1;

	constexpr double pi = 3.1415926535897932384626433832795;
	const double angle = c.alpha * pi / 180.0;
	c.sinA = std::sin(angle);
	c.cosA = std::cos(angle);
	c.grav_x = c.density_x = c.sinA;
	c.grav_y = c.density_y = c.cosA;

	const double steps = std::round(1.0 / c.tau);
	state.steps_per_time_unit = static_cast<unsigned int>(steps);

	std::cout << "N=" << c.N << std::endl;
	if (c.N <= 1024) { cout << "data is too small" << endl; return; }
}


void copy_fields_to_device(const Configuration& c, const Arrays& h, Arrays& d)
{
	copyArrayFromHostToDevice({ d.T, d.T0 }, h.T, c.N);
	copyArrayFromHostToDevice({ d.p, d.p0 }, h.p, c.N);
	copyArrayFromHostToDevice({ d.ux, d.vx }, h.vx, c.N);
	copyArrayFromHostToDevice({ d.uy, d.vy }, h.vy, c.N);
	copyArrayFromHostToDevice({ d.uz, d.vz }, h.vz, c.N);
	copyArrayFromHostToDevice({ d.C, d.C0 }, h.C, c.N);
	copyArrayFromHostToDevice({ d.C2, d.C20 }, h.C2, c.N);
}

void copy_fields_to_host(const Configuration& c, Arrays& h, const Arrays& d, int &already_copied)
{
	if (already_copied) return;
	cudaMemcpy(h.T, d.T, c.Nbytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h.p, d.p, c.Nbytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h.vx, d.vx, c.Nbytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h.vy, d.vy, c.Nbytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h.vz, d.vz, c.Nbytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h.C, d.C, c.Nbytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h.C2, d.C2, c.Nbytes, cudaMemcpyDeviceToHost);
	already_copied = 1;
}

void init_fields(const Configuration& c, Arrays& h, Arrays& d)
{
	std::fill_n(h.T, c.N, 0.0);
	std::fill_n(h.p, c.N, 0.0);
	std::fill_n(h.vx, c.N, 0.0);
	std::fill_n(h.vy, c.N, 0.0);
	std::fill_n(h.vz, c.N, 0.0);
	std::fill_n(h.C, c.N, 0.0);
	std::fill_n(h.C2, c.N, 0.0);

	constexpr double pi = 3.1415926535897932384626433832795;
	for (unsigned int k = 0; k <= c.nz; ++k)
	{
		for (unsigned int j = 0; j <= c.ny; ++j)
		{
			for (unsigned int i = 0; i <= c.nx; ++i)
			{
				const unsigned int q = i + c.offset * j + c.offset2 * k;
				h.C2[q] = 1e-5 * std::sin(2.0 * pi * i * c.hx / c.Lx) *std::sin(2.0 * pi * k * c.hz / c.Lz) *	(1.0 - c.hy * j) * c.psi2;
			}
		}
	}

	copy_fields_to_device(c, h, d);
}

void init_fields_conductive_state(const Configuration& c, Arrays& h, Arrays& d)
{
	std::fill_n(h.T, c.N, 0.0);
	std::fill_n(h.p, c.N, 0.0);
	std::fill_n(h.vx, c.N, 0.0);
	std::fill_n(h.vy, c.N, 0.0);
	std::fill_n(h.vz, c.N, 0.0);
	std::fill_n(h.C, c.N, 0.0);
	std::fill_n(h.C2, c.N, 0.0);

	constexpr double pi = 3.1415926535897932384626433832795;
	constexpr double eps = 1e-5;

	for (unsigned int k = 0; k <= c.nz; ++k)
	{
		for (unsigned int j = 0; j <= c.ny; ++j)
		{
			for (unsigned int i = 0; i <= c.nx; ++i)
			{
				const unsigned int q = i + c.offset * j + c.offset2 * k;

				const double x = i * c.hx;
				const double y = j * c.hy;

				// Проводящее состояние
				h.T[q] = 1.0 - y / c.Ly;
				h.p[q] = 0.0;  // давление установит проекционный шаг
				h.vx[q] = 0.0;
				h.vy[q] = 0.0;
				h.vz[q] = 0.0;

				// Соретовы проводящие профили, среднее по слою равно нулю
				h.C[q] = c.psi1 * c.Q * (y - 0.5 * c.Ly);
				h.C2[q] = c.psi2 * c.Q * (y - 0.5 * c.Ly)

					// Малое возмущение критической 2D-моды kx = pi, kz = 0
					+ eps * std::sin(2.0 * pi * x / c.Lx)
					* std::cos(pi * y / c.Ly);
			}
		}
	}

	copy_fields_to_device(c, h, d);
}
