
#ifndef __Timer__
#define __Timer__

#include <ctime>
#include <vector>

float getCurrentTime();
uint64_t getTimeInUs();
void printTimes(std::vector<float>& vec, bool display = false);

#endif
