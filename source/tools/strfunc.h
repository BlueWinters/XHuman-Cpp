
#ifndef __StrFunc__
#define __StrFunc__

#include <vector>
#include <string>


void splitByChar(const std::string& str, std::vector<std::string>& container, char delim = ' ');
void splitToFloat(const std::string& str, std::vector<float>& container, char delim = ' ');
void splitByString(const std::string& str, std::vector<std::string>& container, const std::string& delims = " ");
std::string formatString(const std::string format, ...);
std::string subString(const std::string& str, int head, int tail);

bool endsWith(const std::string& str, const std::string& suffix);
bool startsWith(const std::string& str, const std::string& prefix);
void joinWith(const std::string& spliter, std::vector<std::string>);

std::string formatString2CharArray(std::string& str, int line_size);
void fromMemory(void* data, int size, std::string& str, const std::string& spliter);

#endif