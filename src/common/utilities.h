#pragma once

#include <string>
#include <windows.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <algorithm>
#include <array>
#include <vector>
#include <map>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <conio.h> // Windows-specific
#include "types.h"


using std::string;
using std::ifstream;
using std::stringstream;
using std::istringstream;
using std::ostringstream;
using std::cout;
using std::endl;


struct RUN_STATE
{
	std::atomic<int> stop_signal{ 0 };
	int compute_mode = 0;
	int call_i = 0, call_max = 1000;
	int read_recovery = 0, full_reset = 0;
	int divice_already_copied = 0;
	double timeq = 0, time_sec = 0;
	double timeq_max = 10000, timeq_min = 0;
	double freq_fields = 10, freq_metrics = 1;
	size_t iter = 0, steps_per_time_unit = 0;
	std::string note = "";
	int test = 0;

	void next_iter(double dt = 0)
	{
		timeq += dt;
		iter++;
		divice_already_copied = 0;
	}


	bool every(int n) { return iter % n == 0; }
	bool every_time(double tau, double t) { int tt = (int)(ceil(t / tau)); return iter % (tt) == 0; }
	void reset()
	{
		iter = 0;
		timeq = 0;
		stop_signal = 0;
		divice_already_copied = 0;
	}


	void check_stop_by_click(int s, const char stop_key = 's')
	{
		if (_kbhit())
		{
			char key = _getch();
			if (key == stop_key)
			{
				stop_signal = s;
				std::cout << "SIGNAL: " << stop_signal << std::endl;
			}

		}
	}
};

struct ReadingFile
{
private:
	ifstream read;
	string buffer;
	string file_name;
	bool opened = false;
	std::map<string, string> values;

	static string trim(string value)
	{
		const auto first = value.find_first_not_of(" \t\r\n");
		if (first == string::npos) return {};
		const auto last = value.find_last_not_of(" \t\r\n");
		return value.substr(first, last - first + 1);
	}

	static string normalized_key(string key)
	{
		key = trim(key);
		std::transform(key.begin(), key.end(), key.begin(),
			[](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
		return key;
	}

	void parse_buffer()
	{
		values.clear();
		istringstream lines(buffer);
		string line;
		unsigned int line_number = 0;

		while (std::getline(lines, line))
		{
			++line_number;
			if (line_number == 1 && line.size() >= 3 &&
				static_cast<unsigned char>(line[0]) == 0xEF &&
				static_cast<unsigned char>(line[1]) == 0xBB &&
				static_cast<unsigned char>(line[2]) == 0xBF)
				line.erase(0, 3);

			const auto comment = line.find('#');
			if (comment != string::npos) line.erase(comment);
			line = trim(line);
			if (line.empty()) continue;

			string key;
			string value;
			const auto separator = line.find('=');
			if (separator != string::npos)
			{
				key = normalized_key(line.substr(0, separator));
				value = trim(line.substr(separator + 1));
			}
			else
			{
				istringstream tokens(line);
				tokens >> key;
				string remainder;
				std::getline(tokens, remainder);
				key = normalized_key(key);
				value = trim(remainder);
			}

			if (key.empty())
				throw std::runtime_error("Malformed parameter line " + std::to_string(line_number));
			if (!values.emplace(key, value).second)
				throw std::runtime_error("Duplicate parameter \"" + key + "\"");
		}
	}

	template <typename T>
	static bool parse_value(const string& text, T& result)
	{
		istringstream stream(text);
		if (!(stream >> result)) return false;
		string trailing;
		return !(stream >> trailing);
	}

public:
	explicit ReadingFile(const string& name)
	{
		open_file(name);
	}
	ReadingFile() = default;

	void open_file(const string& name) {
		file_name = name;
		read.close();
		read.clear();
		read.open(file_name.c_str());
		opened = read.good();

		if (opened) {
			cout << endl << "the parameter file \"" << file_name << "\" has been read " << endl << endl;
#ifdef LoggingON
			Log << endl << "the parameter file \"" << file_name << "\" has been read " << endl << endl;
#endif // LoggingON

			ostringstream oss;
			oss << read.rdbuf();
			buffer = oss.str();
			parse_buffer();
		}
		else {
			cout << "the parameter file has been not found, default parameters will be initialized " << endl;
#ifdef LoggingON
			Log << "the parameter file has been not found, default parameters will be initialized " << endl;
#endif // LoggingON
			buffer = "";
			values.clear();
		}
	}

	template <typename T>
	int reading(T& var, string parameter_name, T def_var = 0, T min = 0, T max = 0) {
		parameter_name = normalized_key(parameter_name);
		const auto it = values.find(parameter_name);
		const int ret = it != values.end() && parse_value(it->second, var);
		if (!ret)
			var = def_var;

		if (min != max) {
			if (var > max || var < min)
			{
				cout << "Warning: \"" + parameter_name + "\" should be within this range" << endl;
				var = def_var;
			}
		}


#ifdef LoggingON
		Log << parameter_name << "= " << var << endl;
#endif // LoggingON

		return ret; //return 1 if read
	}

	template <typename T>
	T required(string parameter_name) const
	{
		parameter_name = normalized_key(parameter_name);
		if (!opened)
			throw std::runtime_error("Required parameter file \"" + file_name + "\" was not found");

		const auto it = values.find(parameter_name);
		if (it == values.end())
			throw std::runtime_error("Required parameter \"" + parameter_name + "\" is missing");

		T result{};
		if (!parse_value(it->second, result))
			throw std::runtime_error("Invalid value for parameter \"" + parameter_name +
				"\": \"" + it->second + "\"");
		return result;
	}

	void reading_string(string& var, string parameter_name, const string& def_var) {
		parameter_name = normalized_key(parameter_name);
		const auto it = values.find(parameter_name);
		var = it == values.end() || it->second.empty() ? def_var : it->second;
#ifdef LoggingON
		Log << parameter_name << "= " << var << endl;
#endif // LoggingON
	}

	bool reading_string(string parameter_name) {
		parameter_name = normalized_key(parameter_name);
		const bool found = values.find(parameter_name) != values.end();
#ifdef LoggingON
		Log << parameter_name << "= " << found << endl;
#endif // LoggingON
		return found;
	}
};

struct FuncTimer
{
private:
	std::map <std::string, double> timer, t1, t2;
	std::string active;
	double dt, total;
public:
	FuncTimer()
	{
		dt = total = 0;
	}
	void start(std::string s)
	{
		active = s;
		t1[s] = clock();
	}
	void end(std::string s)
	{
		if (t1.find(s) == t1.end())
		{
			std::cout << s + " trigger not started" << std::endl;
			return;
		}
		else
		{
			t2[s] = clock();
			dt = (t2[s] - t1[s]) / CLOCKS_PER_SEC;
			//if (s == active)
			{
				timer[s] += dt;
			}
		}
	}

	double get(std::string s)
	{
		//get the time measured 
		if (t1.find(s) == t1.end())
		{
			std::cout << s + " trigger not started" << std::endl;
			return 0.0;
		}
		else
		{
			return timer[s];
		}
	}
	double update_and_get(std::string s)
	{
		// update (apply end now) and get
		if (t1.find(s) == t1.end())
		{
			std::cout << s + " trigger not started" << std::endl;
			return 0.0;
		}
		else
		{
			t2[s] = clock();
			dt = (t2[s] - t1[s]) / CLOCKS_PER_SEC;
			return timer[s] + dt;
		}
	}
	double get_last_diff(std::string s)
	{
		//get the difference (dt) between last end and start: dt = end-start
		if (t1.find(s) == t1.end())
		{
			std::cout << s + " trigger not started" << std::endl;
			return 0.0;
		}
		else
		{
			t2[s] = clock();
			dt = (t2[s] - t1[s]) / CLOCKS_PER_SEC;
			return dt = (t2[s] - t1[s]) / CLOCKS_PER_SEC;
		}
	}

	std::string get_info()
	{
		int n = int(timer.size());
		std::ostringstream oss;
		oss << "Calculation time in seconds. Number of cases: " << n << ".\n";

		for (auto& it : timer)
		{
			oss << it.first << ": " << it.second << std::endl;
		}
		return oss.str();
	}
	void show_info()
	{
		std::cout << get_info() << std::endl;
	}
	void write_info(std::string path = "timer_report.dat")
	{
		std::ofstream file(path);
		file << get_info() << std::endl;
	}
};

struct Checker
{
	enum class ExitType { Relative, Absolute, NoExit };
	static constexpr size_t history_size = 3;

	ExitType type;
	std::string name;
	std::array<double, history_size> values{};
	std::array<double, history_size> times{};
	double* bound_value = nullptr;
	double* bound_time = nullptr;
	double last_time = 0;
	size_t current_index = 0;
	size_t previous_index = 0;
	size_t older_index = 0;
	size_t sample_count = 0;
	double eps_default = 1e-6;
	double relative = 1, absolute = 1;
	double signed_difference = 1, signed_relative = 1;
	double log_ratio = 1, log_rate = 1;
	double derivative = 0, second_derivative = 0;
	double time_step = 1;
	size_t i_check = 0, check_limit = 20;
	bool ready_to_exit = false;

	Checker(double* ptr, double* t_ptr, ExitType type_ = ExitType::Relative, std::string name_ = "Checker", double eps = -1)
	{
		bound_value = ptr;
		bound_time = t_ptr;
		type = type_;
		name = name_;

		if (eps > 0) eps_default = eps;
	}

	bool update()
	{
		const double new_value = *bound_value;
		const double new_time = *bound_time;

		if (sample_count > 0 && last_time == new_time)
			return false;

		current_index = sample_count % history_size;
		previous_index = (current_index + history_size - 1) % history_size;
		older_index = (current_index + history_size - 2) % history_size;

		values[current_index] = new_value;
		times[current_index] = new_time;

		sample_count++;
		last_time = new_time;

		if (sample_count < history_size)
			return false;

		time_step = new_time - times[previous_index];
		const double previous_value = values[previous_index];
		const double older_value = values[older_index];

		signed_difference = new_value - previous_value;
		signed_relative = signed_difference / previous_value;
		absolute = std::abs(signed_difference);
		relative = std::abs(signed_relative);


		derivative = signed_difference / time_step;
		second_derivative = (new_value - 2.0 * previous_value + older_value) / (time_step * time_step);

		log_ratio = std::log(new_value / previous_value);
		log_rate = log_ratio / time_step;

		return true;
	}

	void update(double& val, double& t)
	{
		bound_value = &val;
		bound_time = &t;
		this->update();
	}

	bool check(double eps = -1)
	{
		// Не проверяем повторно старое значение и не проверяем
		// до заполнения всех трёх элементов.
		if (!update())
			return ready_to_exit;

		if (eps < 0)
			eps = eps_default;

		bool condition = false;
		if (type == ExitType::Relative)	condition = relative < eps;		
		if (type == ExitType::Absolute)	condition = absolute < eps;		
		if (type == ExitType::NoExit)	condition = false;		


		if (condition)
		{
			i_check++;
			cout << name << ", i=" << i_check << endl;
		}
		else
		{
			i_check = 0;
		}

		ready_to_exit = i_check >= check_limit;
		return ready_to_exit;
	}

	double estimate_finish(double eps = -1) const
	{
		if (eps < 0)
			eps = eps_default;

		const double e2 = (values[current_index] - values[previous_index]) / values[previous_index];
		const double e1 = (values[previous_index] - values[older_index]) / values[older_index];

		return times[current_index] + time_step * (eps - e2) / (e2 - e1);
	}

	bool operator()()
	{
		return ready_to_exit;
	}

	//bool check_long_term();
};

struct Backup
{
	std::string backup_name[2];
	bool second_backup;

	Backup(std::string name = "recovery", bool second = false)
	{
		backup_name[0] = name + ".txt";
		backup_name[1] = name + "2.txt";
		second_backup = second;
	}
	
	void save(size_t iter, double time, int call_i, Configuration& c,
		std::vector<double*> fields = {}, std::string head = "")
	{
		std::cout << "start writing recovery file" << std::endl;

		auto write_file = [&](const std::string& file_name)
		{
			std::ofstream write(file_name);
			write << "x, y, z, " + head << endl;
			write << "iter,time,call_i,Ra= "
				<< iter << " " << time << " " << call_i << " " << c.Ra << endl;
			write << std::setprecision(std::numeric_limits<double>::max_digits10);

			for (unsigned int k = 0; k <= c.nz; ++k)
			{
				for (unsigned int j = 0; j <= c.ny; ++j)
				{
					for (unsigned int i = 0; i <= c.nx; ++i)
					{
						const unsigned int l = i + c.offset * j + c.offset2 * k;
						write << i * c.hx << " " << j * c.hy << " " << k * c.hz << " ";
						for (double* field : fields)
							write << field[l] << " ";
						write << endl;
					}
				}
			}
		};

		write_file(backup_name[0]);
		if (second_backup)
			write_file(backup_name[1]);

		std::cout << "end writing recovery file" << std::endl;
	}

	void save_atomic(size_t iter, double time, int call_i, Configuration& c,
		std::vector<double*> fields = {}, std::string head = "")
	{
		std::cout << "start writing recovery file" << std::endl;

		const std::string temporary_name = backup_name[0] + ".tmp";
		std::ofstream write(temporary_name);
		if (!write.is_open())
		{
			std::cout << "could not open temporary recovery file" << endl;
			return;
		}

		write << "x, y, z, " + head << endl;
		write << "iter,time,call_i,Ra= "
			<< iter << " " << time << " " << call_i << " " << c.Ra << endl;
		write << std::setprecision(std::numeric_limits<double>::max_digits10);

		for (unsigned int k = 0; k <= c.nz; ++k)
		{
			for (unsigned int j = 0; j <= c.ny; ++j)
			{
				for (unsigned int i = 0; i <= c.nx; ++i)
				{
					const unsigned int l = i + c.offset * j + c.offset2 * k;
					write << i * c.hx << " " << j * c.hy << " " << k * c.hz << " ";
					for (double* field : fields)
						write << field[l] << " ";
					write << endl;
				}
			}
		}

		write.close();
		if (!write.good())
		{
			std::cout << "could not write temporary recovery file" << endl;
			return;
		}

		bool renamed = false;
		#ifdef _WIN32
			renamed = MoveFileExA(temporary_name.c_str(), backup_name[0].c_str(),
			MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
		#else
			renamed = std::rename(temporary_name.c_str(), backup_name[0].c_str()) == 0;
		#endif

		if (!renamed)
		{
			std::cout << "could not replace recovery file" << endl;
			return;
		}

		std::cout << "end writing recovery file" << std::endl;
	}

	int read(size_t& iter, double& time, int& call_i, Configuration& c,
		std::vector<double*> fields = {}, bool read_state = true)
	{
		std::ifstream read;
		if (!second_backup || check2files(backup_name[0], backup_name[1]))
			read.open(backup_name[0]);
		else
			read.open(backup_name[1]);

		if (!read.is_open())
		{
			std::cout << "no recovery file" << endl;
			return 1;
		}

		string line, label;
		getline(read, line); // header
		getline(read, line); // state

		if (read_state)
		{
			stringstream state(line);
			if (!(state >> label >> iter >> time >> call_i >> c.Ra))
				return 1;
		}

		for (unsigned int k = 0; k <= c.nz; ++k)
		{
			for (unsigned int j = 0; j <= c.ny; ++j)
			{
				for (unsigned int i = 0; i <= c.nx; ++i)
				{
					if (!getline(read, line))
						return 1;

					stringstream values(line);
					double x, y, z;
					values >> x >> y >> z;

					const unsigned int l = i + c.offset * j + c.offset2 * k;
					for (double* field : fields)
					{
						if (!(values >> field[l]))
							return 1;
					}
				}
			}
		}
		return 0;
	}

	void read_fields_only(Configuration& c, std::vector<double*> fields = {},	unsigned int skip_lines = 0)
	{
		std::ifstream read;
		if (!second_backup || check2files(backup_name[0], backup_name[1]))
			read.open(backup_name[0]);
		else
			read.open(backup_name[1]);

		string line;
		for (unsigned int q = 0; q < skip_lines; ++q)
			getline(read, line);

		for (unsigned int k = 0; k <= c.nz; ++k)
		{
			for (unsigned int j = 0; j <= c.ny; ++j)
			{
				for (unsigned int i = 0; i <= c.nx; ++i)
				{
					getline(read, line);
					stringstream values(line);
					double x, y, z;
					values >> x >> y >> z;

					const unsigned int l = i + c.offset * j + c.offset2 * k;
					for (double* field : fields)
						values >> field[l];
				}
			}
		}
	}

	bool check2files(const std::string& file_path1, const std::string& file_path2)
	{
		std::ifstream file1(file_path1, std::ios::binary | std::ios::ate);
		std::ifstream file2(file_path2, std::ios::binary | std::ios::ate);
		if (!file2.is_open())
			return true;
		return file1.tellg() >= file2.tellg();
	}

};


size_t allocate_host_arrays(std::vector<double**> v, size_t N)
{
	size_t Nbytes = sizeof(double) * N;
	size_t total_bytes = 0;
	for (auto it : v)
	{
		*it = new double[N];
		memset(*it, 0, Nbytes);
		total_bytes += Nbytes;
	}
	return total_bytes;
}


void pauseWithLineInfo(const std::string& file, int line)
{
	std::cout << "Paused at file: " << file << ", line: " << line << std::endl;
	std::cout << "Press Enter to continue...";
	std::cin.ignore();
}
#define pause pauseWithLineInfo(__FILE__, __LINE__);





void deleteFilesInDirectory(const std::wstring& directoryPath) {
	WIN32_FIND_DATAW findFileData;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	// Create a search path for all files in the directory
	std::wstring searchPath = directoryPath + L"\\*";

	// Find the first file in the directory
	hFind = FindFirstFileW(searchPath.c_str(), &findFileData);
	if (hFind == INVALID_HANDLE_VALUE) {
		std::wcerr << L"Error opening directory: " << directoryPath << std::endl;
		return;
	}

	do {
		// Skip the special directories "." and ".."
		if (wcscmp(findFileData.cFileName, L".") == 0 || wcscmp(findFileData.cFileName, L"..") == 0) {
			continue;
		}

		// Construct the full file path
		std::wstring filePath = directoryPath + L"\\" + findFileData.cFileName;

		// Check if the entry is a regular file
		if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
			// Delete the file
			if (DeleteFileW(filePath.c_str()))
			{
				//std::wcout << L"Deleted file: " << filePath << std::endl;
			}
			else {
				std::wcerr << L"Error deleting file: " << filePath << std::endl;
			}
		}
	} while (FindNextFileW(hFind, &findFileData) != 0);

	// Close the search handle
	FindClose(hFind);
}
std::streamoff fileSize(const std::string& filePath) {
	std::ifstream file(filePath, std::ios::binary | std::ios::ate);
	if (!file.is_open()) { std::cerr << "Error!" << std::endl;        return false; }
	std::streampos bytes = file.tellg();
	return bytes.operator std::streamoff(); 	//file size in bytes
}

void addFolder(std::string fname, std::string& global)
{
#ifdef __linux__ 
	string str = "mkdir -p " + fname + "/";
	global = fname + "/";
	system(str.c_str());
#endif

#ifdef _WIN32
	CreateDirectoryA(fname.c_str(), NULL);
	global = fname + "//";
#endif
}

std::string _path(std::string fname)
{
#ifdef __linux__ 
	return fname + "/";
#endif

#ifdef _WIN32
	return fname + "//";
#endif
}





std::string _str(double n, int pres = 5)
{
	std::stringstream ss;
	ss << std::setprecision(pres) << n;
	return ss.str();
}
std::string _str(int n, int pres = 5)
{
	std::stringstream ss;
	ss << std::setprecision(pres) << n;
	return ss.str();
}
std::string _str(size_t n, int pres = 5)
{
	std::stringstream ss;
	ss << std::setprecision(pres) << n;
	return ss.str();
}

