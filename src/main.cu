#pragma warning(disable:4101)
#pragma warning(disable:4018)  
#pragma warning(disable:4389)  
#pragma warning(disable:4244)
#include "cuda_runtime.h"
#include <iostream> 
#include <string> 
#include <fstream> 
#include <iomanip> 
#include <cmath>
#include <algorithm> 
#include <vector>
#ifdef _WIN32
#include "windows.h"
#endif

#include "common/types.h"
#include "common/math.h"
#include "common/utilities.h"
#include "common/utilities_cuda.h"
#include "common/init.h"

//GPU functions and variables
__constant__ Configuration dev;
Configuration host;

#include "cfd/diagnostics.h"
#include "cfd/solver.h"
#include "cfd/solver_output.h"

RUN_STATE run;
FuncTimer ftimer;


int main(int argc, char** argv) 
{
	const bool smoke_test = argc == 2 && std::string(argv[1]) == "--smoke-test";
	init_parameters(host, run);



	GPU_ gpu(0);
	Arrays h{}, d{};
	StatValues stats{};


	h.N = d.N = host.N;
	h.Nbytes = d.Nbytes = host.Nbytes;
	h.total_bytes = allocate_host_arrays(	{ &h.T, &h.p, &h.vx, &h.vy, &h.vz, &h.C, &h.C2 }, host.N);
	d.total_bytes = allocate_device_arrays(	{ &d.T, &d.T0, &d.p, &d.p0, &d.ux, &d.uy, &d.uz,  &d.vx, &d.vy, &d.vz, &d.C, &d.C0, &d.C2, &d.C20 }, host.N);
	gpu.gpu_mem_allocated = d.total_bytes;


	//ofstream is a class to write data in a file 
	std::ofstream Ra_tab;
	std::ofstream integrals;

	Backup recovery("recovery");
	const std::vector<double*> recovery_fields{	h.vx, h.vy, h.vz, h.p, h.T, h.C, h.C2, };
	const std::vector<double*> output_fields  {	h.vx, h.vy, h.vz, h.p, h.T, h.C, h.C2, };
	

	const double Ra_from_parameters = host.Ra;
	const int initial_call_i = run.call_i;
	bool recovered = false;

	if (run.read_recovery)
	{
		recovered =	recovery.read(run.iter, run.timeq, run.call_i, host, recovery_fields) == 0;
		if (recovered)
		{
			copy_fields_to_device(host, h, d);
			cout << "continue from t= " << run.timeq << endl;
		}
		else
		{
			std::cerr << "Recovery file could not be read. Starting from initial fields."	<< std::endl;
		}
	}
	if (!recovered)
	{
		host.Ra = Ra_from_parameters;
		run.call_i = initial_call_i;
		run.reset();
		//init_fields(host, h, d);
		init_fields_conductive_state(host, h, d);
		Ra_tab.open("Ra.dat");
		integrals.open("integrals.dat");
		write_ra_row(Ra_tab, host, run, stats, OutputMetrics{}, 0.0, true);
	}
	else
	{
		Ra_tab.open("Ra.dat", std::ofstream::app);
		integrals.open("integrals.dat", std::ofstream::app);
	}


	Solver solver(host, d);

	for (bool first_run = true; run.call_i < run.call_max; ++run.call_i)
	{
		if (run.stop_signal == -1) break;
		if (first_run)	first_run = false;
		else
			host.Ra += host.incr_parameter;

		//just printing parameters from GPU to be confident they are passed correctly
		cudaMemcpyToSymbol(dev, &host, sizeof(Configuration), 0, cudaMemcpyHostToDevice);
		device_parameter_check << <1, 1 >> > ();
		cudaDeviceSynchronize();

		ftimer.start("main");
		while (run.stop_signal == 0)
		{
			run.check_stop_by_click(1, 's');
			run.check_stop_by_click(-1, 'q');

			run.next_iter(host.tau);
			if (run.timeq > run.timeq_max) run.stop_signal = 1;


			solver.proceed_binary_fixed_temperature();

			//metrics output
			if (run.every_time(host.tau, run.freq_metrics) || run.iter == 1)
			{
				copy_fields_to_host(host, h, d, run.divice_already_copied);

				
				OutputMetrics metrics = collect_output_metrics(host, h, stats);
				print_metrics(cout, host, run, stats, metrics);
				const double elapsed_seconds = ftimer.update_and_get("main");

				write_integrals_row(integrals, host, run, stats, metrics, elapsed_seconds);

				//if ((run.timeq > 1500 && abs(stats.Ek - stats.Ek_old) < 1e-8 && host.Ra == 5000) || (run.timeq > 1000 && abs(stats.Ek - stats.Ek_old) < 1e-7 && host.Ra < 5000)) {
				if ((run.timeq > run.timeq_min && (metrics.Ek_change_relative < 1e-7 || metrics.Ek_change_abs < 1e-12)))
				{
					run.stop_signal = 1;
				}
				stats.Ek_old = stats.Ek;
			}

			//recovery
			if (run.timeq > 10 && run.every_time(host.tau, 10.0))
			{
				copy_fields_to_host(host, h, d, run.divice_already_copied);
				recovery.save_atomic(run.iter, run.timeq, run.call_i, host, recovery_fields, "vx, vy, vz, p, T, C1, C2");
			}

			//field output
			if (run.iter == 1 || run.every_time(host.tau, run.freq_fields))
			{
				copy_fields_to_host(host, h, d, run.divice_already_copied);
				std::string label = _str(host.Ra) + " " + _str(run.timeq);
				std::string extra_line = "iter,time,call_i,Ra= " + _str(run.iter) + " " + _str(run.timeq) + " " + _str(run.call_i) + " " + _str(host.Ra);

				write_fields_section("fields/section/", label, host, output_fields, "vx, vy, vz, p, T, C1, C2");
				write_fields_section("fields/section2/", label, host, output_fields, "vx, vy, vz, p, T, C1, C2", 100000);
				write_fields("fields/full/", label, host, output_fields, "vx, vy, vz, p, T, C1, C2", extra_line);
			}


			//exit time loop
			if (run.stop_signal != 0)
			{
				std::string label = _str(host.Ra) + " " + _str(run.timeq);
				std::string extra_line = "iter,time,call_i,Ra= " + _str(run.iter) + " " + _str(run.timeq) + " " + _str(run.call_i) + " " + _str(host.Ra);
				OutputMetrics metrics = collect_output_metrics(host, h, stats);
				const double elapsed_seconds = ftimer.update_and_get("main");

				write_ra_row(Ra_tab, host, run, stats, metrics, elapsed_seconds, false);
				write_fields("fields/final/", label, host, output_fields, "vx, vy, vz, p, T, C1, C2", extra_line);

				if (run.stop_signal == 1) run.reset();
				break;
			}

			if (smoke_test)
			{
				const cudaError_t smoke_status = cudaDeviceSynchronize();
				if (smoke_status != cudaSuccess)
				{
					std::cerr << "CUDA smoke test failed: " << cudaGetErrorString(smoke_status) << std::endl;
					return EXIT_FAILURE;
				}
				std::cout << "CUDA smoke test completed one time step successfully" << std::endl;
				run.stop_signal = -1;
				break;
			}

		}  //end main time loop

	} // end Ra++ loop



	return 0;
}
