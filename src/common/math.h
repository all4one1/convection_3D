#pragma once
#include <cmath>
#include <algorithm> 
#define Pi 3.1415926535897932384626433832795

inline double sum_signed(double* f, unsigned int N)
{
	double s = 0;

	for (unsigned int i = 0; i < N; i++)
		s += f[i];
	return s;
}
inline double sum_abs(double* f, unsigned int N)
{
	double s = 0;

	for (unsigned int i = 0; i < N; i++)
		s += abs(f[i]);
	return s;
}
inline double max_signed(double* f, unsigned int N)
{
	double m = f[0];

	for (unsigned int i = 0; i < N; i++)
	{
		if (f[i] > m) m = f[i];
	}
	return m;
}
inline double min_signed(double* f, unsigned int N)
{
	double m = f[0];

	for (unsigned int i = 0; i < N; i++)
	{
		if (f[i] < m) m = f[i];
	}
	return m;
}
inline double max_abs(double* f, unsigned int N)
{
	double m = f[0];

	for (unsigned int i = 0; i < N; i++)
	{
		if (abs(f[i]) > m) m = abs(f[i]);
	}
	return m;
}
inline double min_abs(double* f, unsigned int n)
{
	double min = abs(f[0]);

	for (unsigned int i = 0; i < n; i++) {
		if (abs(f[i]) < min)
		{
			min = abs(f[i]);
		}
	}
	return min;
}

