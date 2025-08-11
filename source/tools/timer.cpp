
#include <stdint.h>
#include <iostream>
#include "timer.h"
#include "strfunc.h"


#ifdef _WIN32
#include <windows.h>
float getCurrentTime()
{
	LARGE_INTEGER freq;
	LARGE_INTEGER pc;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&pc);
	return pc.QuadPart * 1000.0 / freq.QuadPart;
}
#else
#include <sys/time.h>
float getCurrentTime()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif

#if defined(_MSC_VER)
#include <windows.h>
uint64_t getTimeInUs() 
{
	uint64_t time;
	LARGE_INTEGER now, freq;
	QueryPerformanceCounter(&now);
	QueryPerformanceFrequency(&freq);
	uint64_t sec = now.QuadPart / freq.QuadPart;
	uint64_t usec = (now.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
	time = sec * 1000000 + usec;
	return time;
}
#else
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
uint64_t getTimeInUs()
{
	struct timeval tv;
	gettimeofday(&tv, nullptr);
	return static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}
#endif


#define MaxMacro(a, b) (((a) > (b)) ? (a) : (b))
#define MinMacro(a, b) (((a) < (b)) ? (a) : (b))

void printTimes(std::vector<float>& vec, bool display)
{
	float max_time = vec[0];
	float min_time = vec[0];
	float avg_time = 0;
	for (int n = 0; n < vec.size(); n++)
	{
		float time = vec[n];
		max_time = MaxMacro(max_time, time);
		min_time = MinMacro(min_time, time);
		if (display)
		{
			std::string line = formatString("\ttime: %-8.2fms", time);
			std::cout << line << std::endl;
		}
		avg_time += time;
	}

	std::string line = formatString("  min=%.2f ms, max=%.2f ms, avg=%.2f ms",
		min_time, max_time, avg_time / vec.size());
	std::cout << line << std::endl;
}




