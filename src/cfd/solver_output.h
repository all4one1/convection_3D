#pragma once
#include <fstream>
#include <iomanip>
#include <ostream>
#include <string>
#include <vector>
#include <cmath>
#include "../common/types.h"
#include "../common/math.h"
#include "../common/utilities.h"
#include "diagnostics.h"

struct OutputMetrics
{
	double vxmax = 0;
	double vymax = 0;
	double vzmax = 0;
	double pressure_max = 0;
	double Nu = 0;
	double Sh = 0;
	double Tmax_xz = 0;
	double Txm = 0;
	double Tzm = 0;
	double Vmax_xz = 0;
	double Vxm = 0;
	double Vzm = 0;
	double vy_control = 0;
	double center_vy = 0;
	double center_T = 0;
	double center_C = 0;
	double quarter_vy = 0;
	double quarter_T = 0;
	double quarter_C = 0;
	double Ek_change = 0;
	double Ek_change_abs = 0;
	double Ek_change_relative = 0;
};

inline unsigned int field_index(const Configuration& c, unsigned int i, unsigned int j, unsigned int k)
{
	return i + c.offset * j + c.offset2 * k;
}

inline OutputMetrics collect_output_metrics(const Configuration& c, const Arrays& h, StatValues& stats)
{
	OutputMetrics metrics;

	velocity(c, h.vx, h.vy, h.vz, stats.Ek, stats.Vmax);
	stats.Ek /= c.Lx * c.Ly * c.Lz;

	metrics.vxmax = max_abs(h.vx, c.N);
	metrics.vymax = max_abs(h.vy, c.N);
	metrics.vzmax = max_abs(h.vz, c.N);
	metrics.pressure_max = max_abs(h.p, c.N);

	Nu_y_for_fixed_flux(c, h.T, metrics.Nu);
	Nu_y_for_fixed_flux(c, h.C2, metrics.Sh);

	unsigned int i, k;
	find_max_abs_xz_section_with_index(c, h.T, i, k, metrics.Tmax_xz);
	metrics.Txm = i * c.hx;
	metrics.Tzm = k * c.hz;

	find_max_Vabs_xz_section_with_index(c, h.vx, h.vy, h.vz, i, k, metrics.Vmax_xz);
	metrics.Vxm = i * c.hx;
	metrics.Vzm = k * c.hz;

	stats.Tm = max_abs(h.T, c.N);
	stats.Cm = max_abs(h.C2, c.N);
	angular_momentum(c, h.vx, h.vy, h.vz, stats.AMx, stats.AMy, stats.AMz, stats.AM);

	metrics.vy_control = h.vy[field_index(c, 10, 10, 10)];
	const unsigned int center = field_index(c, c.nx / 2, c.ny / 2, c.nz / 2);
	const unsigned int quarter = field_index(c, c.nx / 4, c.ny / 4, c.nz / 4);
	metrics.center_vy = h.vy[center];
	metrics.center_T = h.T[center];
	metrics.center_C = h.C2[center];
	metrics.quarter_vy = h.vy[quarter];
	metrics.quarter_T = h.T[quarter];
	metrics.quarter_C = h.C2[quarter];

	metrics.Ek_change = stats.Ek - stats.Ek_old;
	metrics.Ek_change_abs = std::abs(metrics.Ek_change);
	metrics.Ek_change_relative = metrics.Ek_change_abs / stats.Ek_old;

	return metrics;
}

inline void print_metrics(std::ostream& output, const Configuration& c, const RUN_STATE& run,
	const StatValues& stats, const OutputMetrics& metrics)
{
	output << std::setprecision(15) << std::fixed << "\n\n\n";
	output << "call_i=" << run.call_i << '\n';
	output << "Ra=" << c.Ra << '\n';
	output << "Vx=" << metrics.vxmax << '\n';
	output << "Vy=" << metrics.vymax << '\n';
	output << "Vz=" << metrics.vzmax << '\n';
	output << "Vy_c=" << metrics.vy_control << '\n';
	output << "p=" << metrics.pressure_max << '\n';
	output << "T=" << stats.Tm << '\n';
	output << "C=" << stats.Cm << '\n';
	output << "Ek= " << stats.Ek << '\n';
	output << "dEk= " << metrics.Ek_change << '\n';
	output << "dEk= " << metrics.Ek_change_relative << '\n';
	output << "t=" << std::round(run.timeq * 10000.0) / 10000.0 << '\n';
}

inline void write_integrals_row(std::ostream& integrals, const Configuration& c, const RUN_STATE& run, const StatValues& stats,
	const OutputMetrics& metrics, double elapsed_seconds)
{
	if (run.iter == 1)
	{
		integrals << "t, Ek, Vmax,  time(min), Vc, Tc, Cc, Vq, Tq, Cq, dEk, dEkRel, AMx, AMy, AMz, AMabs, vxmax, vymax, vzmax, ";
		integrals << "Nu, Sh, Tmax_xm, Txm, Tzm, Vmax_xm, Vxm, Vzm, ";
		integrals << "Ra=" << c.Ra << '\n';
	}

	integrals << std::setprecision(20) << std::fixed;
	integrals << run.timeq << " " << stats.Ek << " " << stats.Vmax << " " << elapsed_seconds / 60 << " ";
	integrals << metrics.center_vy << " " << metrics.center_T << " " << metrics.center_C << " ";
	integrals << metrics.quarter_vy << " " << metrics.quarter_T << " " << metrics.quarter_C << " ";
	integrals << metrics.Ek_change_abs << " " << metrics.Ek_change_relative << " ";
	integrals << stats.AMx << " " << stats.AMy << " " << stats.AMz << " " << stats.AM << " ";
	integrals << metrics.vxmax << " " << metrics.vymax << " " << metrics.vzmax << " ";
	integrals << metrics.Nu << " " << metrics.Sh << " ";
	integrals << metrics.Tmax_xz << " " << metrics.Txm << " " << metrics.Tzm << " ";
	integrals << metrics.Vmax_xz << " " << metrics.Vxm << " " << metrics.Vzm << '\n';

	integrals.flush();
}

inline void write_ra_row(std::ostream& ra_tab, const Configuration& c, const RUN_STATE& run,
	const StatValues& stats, const OutputMetrics& metrics, double elapsed_seconds, bool write_header)
{
	if (write_header)
	{
		ra_tab << "Ra, Ek, Vmax, Vc, time(min), t, AMx, AMy, AMz, AMabs, Nu, Sh, Tmax_xm, Txm, Tzm\n";
		return;
	}

	ra_tab << c.Ra << " " << stats.Ek << " " << stats.Vmax << " " << metrics.center_vy << " " << elapsed_seconds << " " << run.timeq
		<< " " << stats.AMx << " " << stats.AMy << " " << stats.AMz << " " << stats.AM
		<< " " << metrics.Nu << " " << " " << metrics.Sh << " " << metrics.Tmax_xz << " " << metrics.Txm << " " << metrics.Tzm
		<< '\n';
	ra_tab.flush();
}

void write_fields(std::string folder, std::string name, Configuration& c, std::vector<double*> v = {}, std::string head = "", std::string extra_line = "")
{
	std::ofstream w((folder + name + ".txt").c_str());

	w << "x, y, z, " << head << std::endl;

	if (extra_line != "")
		w << extra_line << std::endl;

	double x, y, z;
	for (unsigned int k = 0; k <= c.nz; k++) {
		for (unsigned int j = 0; j <= c.ny; j++) {
			for (unsigned int i = 0; i <= c.nx; i++) {
				unsigned int l = i + c.offset * j + c.offset2 * k;

				x = i * c.hx;
				y = j * c.hy;
				z = k * c.hz;

				w << x << " " << y << " " << z << " ";

				for (auto& it : v)
				{
					w << it[l] << " ";
				}
				w << std::endl;
			}
		}
	}
}

void write_fields_section(std::string root_folder, std::string name, Configuration& c, std::vector<double*> v = {}, std::string head = "", int denom = 2)
{
	std::ofstream xz((root_folder + "/main/ main_section_xz2_" + name + ".txt").c_str());
	std::ofstream xy((root_folder + "/face/ face_section_xy2_" + name + ".txt").c_str());
	std::ofstream yz((root_folder + "/cross/ cross_section_yz2_" + name + ".txt").c_str());


	xy << "x, y, " << head << std::endl;
	for (unsigned int i = 0; i <= c.nx; i++) {
		for (unsigned int j = 0; j <= c.ny; j++) {
			unsigned int l = i + c.offset * j + c.offset2 * (c.nz / denom);
			xy << i * c.hx << " " << j * c.hy << " ";
			for (auto& it : v) {
				xy << it[l] << " ";
			}
			xy << std::endl;
		}
	}

	yz << "z, y, " << head << std::endl;
	for (unsigned int k = 0; k <= c.nz; k++) {
		for (unsigned int j = 0; j <= c.ny; j++) {
			unsigned int l = c.nx / denom + c.offset * j + c.offset2 * k;
			yz << k * c.hz << " " << j * c.hy << " ";
			for (auto& it : v) {
				yz << it[l] << " ";
			}
			yz << std::endl;
		}
	}

	xz << "x, z, " << head << std::endl;
	for (unsigned int i = 0; i <= c.nx; i++) {
		for (unsigned int k = 0; k <= c.nz; k++) {
			unsigned int l = i + c.offset * (c.ny / denom) + c.offset2 * k;
			xz << i * c.hx << " " << k * c.hz << " ";
			for (auto& it : v) {
				xz << it[l] << " ";
			}
			xz << std::endl;
		}
	}
}


