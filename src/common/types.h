#pragma once

#include <cstddef>
#include <type_traits>

enum discretization { FV = 0, FD = 1 };
enum bc_type { closed = 0, periodic = 1, Dirichlet = 2, Neumann = 3, not_boundary = 4 };
enum domain_config { closed_box, open_tube, periodic_cell };
enum field_name {
	field_T, field_vx, field_vy, field_p, field_p_prime, field_vx_prime, field_vy_prime,
	field_C, field_mu, field_C2, field_mu2, field_C3, field_mu3, field_stream, field_omega, field_count
};
//enum enumside { west, east, south, north, front, back, inner };

enum class MathBoundary { Dirichlet, Neumann, periodic, not_boundary, array_value, complex };
enum class Side { center, west, east, south, north, front, back };
enum class Component { x, y, z };


struct Configuration
{
	double hx = 0.0, hy = 0.0, hz = 0.0;
	double Sx = 0.0, Sy = 0.0, Sz = 0.0;
	double Lx = 0.0, Ly = 0.0, Lz = 0.0, dV = 0.0;
	double tau = 0.0, tau_p = 0.0;
	double alpha = 0.0, beta = 0.0, gamma = 0.0;
	double sinA = 0.0, cosA = 1.0, sinB = 0.0, cosB = 1.0, sinG = 0.0, cosG = 1.0;
	double vibr_x = 0.0, vibr_y = 0.0, vibr_z = 0.0;
	double grav_x = 0.0, grav_y = 1.0, grav_z = 0.0;
	double density_x = 0.0, density_y = 1.0, density_z = 0.0;
	double Sc = 0.0, psi = 0.0, Sc11 = 0.0, Sc12 = 0.0, Sc21 = 0.0, Sc22 = 0.0;
	double psi1 = 0.0, psi2 = 0.0, psiS = 0.0, Sc2 = 0.0, Q = 0.0;
	double Ra = 0.0, Rad = 0.0, Rav = 0.0, K = 0.0, Pr = 0.0, Le = 0.0;
	double Re = 0.0, Pe = 0.0, M = 0.0, A = 0.0, Ca = 0.0, Gr = 0.0;
	double incr_parameter = 0.0;
	double convergence_tolerance = 0.0;
	unsigned int dim = 0, nx = 0, ny = 0, nz = 0, N = 0, offset = 0, offset2 = 0;
	std::size_t Nbytes = 0;
	unsigned int heatflux = 0, full_fields = 0;
	bc_type xbc = closed, ybc = closed, zbc = closed;
	domain_config domain = closed_box;
	discretization disc = FD;
};

static_assert(std::is_trivially_copyable<Configuration>::value,	"Configuration must be safe to copy to CUDA constant memory");

struct StatValues
{
	double Ek = 0, Ek_old = 0, Vmax = 0, Tm = 0, Cm = 0, AM = 0, AMx = 0, AMy = 0, AMz = 0;
	double ksi_max = 0, ksi_sum = 0, omega_max = 0, omega_sum = 0, C_sum = 0, C_sum_signed = 0;
	double ksi_incr = 0;
	double NuTop = 0, NuDown = 0, ShrTop = 0, ShrDown = 0;
	double Nu = 0, Shr = 0;
	double Nu2 = 0, Shr2 = 0;
	double Vx = 0, Vy = 0, Vz = 0;
	double Cu = 0, Pe = 0;
	unsigned int n_vortex = 0;
};


struct Arrays
{
	double* p, * p0, * ux, * uy, * uz, * vx, * vy, * vz, * buffer, * buffer2, * rhs;
	double* T, * T0, * C, * C0, * C2, * C20, * C3, * C30, * mu, * mu0;
	double* omega, * omega0, * ksi, * ksi0;
	double* src_x, * src_y, * src_z;
	size_t total_bytes, Nbytes, N;
};

