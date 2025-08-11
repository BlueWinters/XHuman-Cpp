
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <cstdarg>
#include <iomanip>
#include "strfunc.h"


void splitByChar(const std::string& str, std::vector<std::string>& container, char delim /*= ' '*/)
{
	std::istringstream stream(str);
	std::string token;
	while (std::getline(stream, token, delim))
	{
		// TODO: store the token
		container.push_back(token);
	}
}

void splitToFloat(const std::string& str, std::vector<float>& container, char delim /*= ' '*/)
{
	std::istringstream stream(str);
	std::string token;
	float value;
	while (stream.eof() == false)
	{
		// TODO: store the token
		if (stream >> value)
			container.push_back(value);
	}
}

void splitByString(const std::string& str, std::vector<std::string>& container, const std::string& delims /*= " "*/)
{
	std::size_t length = delims.size();
	if (delims.empty() == true)
		return; // invalid sub string

	std::size_t current, previous = 0;
	current = str.find(delims);
	while (current != std::string::npos)
	{
		std::string substr = str.substr(previous, current - previous);
		// TODO: trim the null string and store the token
		if (previous < current)
			container.push_back(substr);
		previous = current + length;
		current = str.find(delims, previous);
	}

	// TODO: trim the null string and store the last token
	if (previous < current)
		container.push_back(str.substr(previous, current - previous));
}

std::string formatString(const std::string format, ...)
{
	va_list args;
	va_start(args, format);
	size_t len = std::vsnprintf(NULL, 0, format.c_str(), args);
	va_end(args);

	std::vector<char> vec(len + 1);
	va_start(args, format);
	std::vsnprintf(&vec[0], len + 1, format.c_str(), args);
	va_end(args);
	return &vec[0];
}

std::string subString(const std::string& str, int head, int tail)
{
	unsigned int size = str.size();
	if (size > 0)
	{
		unsigned int pos_head = (head >= 0) ? head : (size + head);
		unsigned int pose_tail = (tail >= 0) ? tail : (size + tail);
		return str.substr(pos_head, pose_tail);
	}
	else
	{
		return std::string("");
	}
}

bool endsWith(const std::string& str, const std::string& suffix)
{
	auto str_size = str.size();
	auto sfx_size = suffix.size();
	if (str_size > sfx_size && sfx_size > 0)
	{
		return static_cast<bool>(str.compare(str_size - sfx_size, sfx_size, suffix) == 0);
	}

	return false;
}

bool startsWith(const std::string& str, const std::string& prefix)
{
	auto str_size = str.size();
	auto pfx_size = prefix.size();
	if (str_size > pfx_size && pfx_size > 0)
	{
		return static_cast<bool>(str.compare(0, pfx_size, prefix) == 0);
	}

	return false;
}

std::string formatString2CharArray(std::string& str, int line_size)
{
	auto size = str.size();
	std::string format;
	
	auto group = static_cast<int>(static_cast<float>(size)/line_size + 0.5f);
	for (auto g = 0; g < group; g++)
	{
		auto beg = line_size * g;
		auto end = line_size * (g+1);
		format += '\t';
		format += '\"';
		auto sub = str.substr(beg, line_size);
		format += str.substr(beg, line_size);
		format += '\"';
		format += '\n';
	}
	return format;
}

void fromMemory(void* data, int size, std::string& str, const std::string& spliter)
{
	const int line_size = 32;
	unsigned char* ptr = static_cast<unsigned char*>(data);
	str.reserve((spliter.size() + 3) * size);
	for (int n = 0; n < size; n++)
	{
		unsigned char v = ptr[n];
		str += ((n+0) % line_size == 0) ? "\t" : "";
		str += formatString("0x%02X", v);
		str += spliter;
		str += ((n+1) % line_size == 0) ? "\n" : "";
	}
}



//void main()
//{
//	char str[] = "the quick brown fox jumps over the lazy dog";
//
//	std::vector<std::string> words1;
//	splitByChar(str, words1, ' ');
//	std::copy(words1.begin(), words1.end(),
//		std::ostream_iterator<std::string>(std::cout, "\n"));
//
//	std::vector<std::string> words2;
//	splitByString(str, words2, "the");
//	std::copy(words2.begin(), words2.end(),
//		std::ostream_iterator<std::string>(std::cout, "\n"));
//
//	system("pause");
//}