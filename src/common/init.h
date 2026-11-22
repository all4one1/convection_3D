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

	ReadingFile par("parameters.txt");
	state.read_recovery = par.required<int>("read_recovery");
	state.timeq_min = par.required<double>("timeq_min");
	state.timeq_max = par.required<double>("timeq_max");
	state.freq_fields = par.required<double>("freq_fields");
	par.reading(state.freq_metrics, "freq_metrics", 1.0);
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
	c.hx = c.Lx / static_cast<double>(c.nx - (c.xbc == periodic ? 1 : 0));
	c.hy = c.Ly / static_cast<double>(c.ny);
	c.hz = c.Lz / static_cast<double>(c.nz - (c.zbc == periodic ? 1 : 0));
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
	c.tau_p = 0.25 / (1.0 / (c.hx * c.hx) + 1.0 / (c.hy * c.hy) + 1.0 / (c.hz * c.hz));
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
	// Restart files may contain periodic ghosts from an earlier time level.
	auto refresh_periodic_planes = [&](double* field, bool wrap_x, bool wrap_z)
		{
			if (wrap_x)
				for (unsigned int k = 0; k <= c.nz; ++k)
					for (unsigned int j = 0; j <= c.ny; ++j)
					{
						double* row = field + c.offset * j + c.offset2 * k;
						row[0] = row[c.nx - 1];
						row[c.nx] = row[1];
					}
			if (wrap_z)
			{
				std::copy_n(field + c.offset2 * (c.nz - 1), c.offset2, field);
				std::copy_n(field + c.offset2, c.offset2, field + c.offset2 * c.nz);
			}
		};
	const bool wrap_x = c.xbc == periodic;
	const bool wrap_z = c.zbc == periodic;
	refresh_periodic_planes(h.T, wrap_x, wrap_z);
	refresh_periodic_planes(h.C, wrap_x, wrap_z);
	refresh_periodic_planes(h.C2, wrap_x, wrap_z);
	refresh_periodic_planes(h.p, wrap_x, wrap_z);
	for (double* velocity : { h.vx, h.vy, h.vz })
	{
		// Legacy mixed-boundary restarts may have nonzero velocity on solid faces.
		if (c.xbc == closed)
			for (unsigned int k = 0; k <= c.nz; ++k)
				for (unsigned int j = 0; j <= c.ny; ++j)
				{
					double* row = velocity + c.offset * j + c.offset2 * k;
					row[0] = 0.0;
					row[c.nx] = 0.0;
				}
		if (c.zbc == closed)
		{
			std::fill_n(velocity, c.offset2, 0.0);
			std::fill_n(velocity + c.offset2 * c.nz, c.offset2, 0.0);
		}
		refresh_periodic_planes(velocity, wrap_x, wrap_z);
	}

	copyArrayFromHostToDevice({ d.T, d.T0 }, h.T, c.N);
	copyArrayFromHostToDevice({ d.p, d.p0 }, h.p, c.N);
	copyArrayFromHostToDevice({ d.ux, d.vx }, h.vx, c.N);
	copyArrayFromHostToDevice({ d.uy, d.vy }, h.vy, c.N);
	copyArrayFromHostToDevice({ d.uz, d.vz }, h.vz, c.N);
	copyArrayFromHostToDevice({ d.C, d.C0 }, h.C, c.N);
	copyArrayFromHostToDevice({ d.C2, d.C20 }, h.C2, c.N);
}

void copy_fields_to_host(const Configuration& c, Arrays& h, const Arrays& d, int& already_copied)
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
				h.C2[q] = 1e-5 * std::sin(2.0 * pi * i * c.hx / c.Lx) * std::sin(2.0 * pi * k * c.hz / c.Lz) * (1.0 - c.hy * j) * c.psi2;
			}
		}
	}

	copy_fields_to_device(c, h, d);
}

// Periodic directions have two ghost planes and nx-1 / nz-1 unique points.
void init_fields_conductive_state(const Configuration& c, Arrays& h, Arrays& d)
{
	std::fill_n(h.vx, c.N, 0.0);
	std::fill_n(h.vy, c.N, 0.0);
	std::fill_n(h.vz, c.N, 0.0);
	constexpr double pi = 3.1415926535897932384626433832795;
	for (unsigned int k = 0; k <= c.nz; ++k)
		for (unsigned int j = 0; j <= c.ny; ++j)
			for (unsigned int i = 0; i <= c.nx; ++i)
			{
				const unsigned int q = i + c.offset * j + c.offset2 * k;
				const double x = (static_cast<double>(i) - (c.xbc == periodic ? 1.0 : 0.0)) * c.hx;
				const double z = (static_cast<double>(k) - (c.zbc == periodic ? 1.0 : 0.0)) * c.hz;
				const double y = j * c.hy;
				h.T[q] = c.Q * (y - 0.5 * c.Ly);
				h.C[q] = c.psi1 * h.T[q];
				// seed as in PoF 38, 064104 (2026): 1e-5*sin(2pi x/Lx)*sin(2pi y/Ly)*(1-z)*psi
				h.C2[q] = c.psi2 * h.T[q] + 1e-5 * std::sin(2 * pi * x / c.Lx)
					* std::sin(2 * pi * y / c.Ly) * (1.0 - y) * c.psiS;
				h.p[q] = c.Ra / c.Pr * (1 + c.psiS) * c.Q * (0.5 * y * y - 0.5 * c.Ly * y);
			}
	copy_fields_to_device(c, h, d);
}
