
#ifndef __cvFunc__
#define __cvFunc__

#include <vector>
#include <string>
#include <opencv2\core\mat.hpp>

// only for image
bool formatCVMat2BufferC(const cv::Mat& mat, int& height, int& width, int& channel, unsigned char** buffer);
bool formatCVMat2BufferP(const cv::Mat& mat, int& height, int& width, int& channel, unsigned char** buffer);
bool formatBufferP2Mat(const unsigned char* buffer, int height, int width, int channel, cv::Mat& mat);
bool formatBufferC2Mat(const unsigned char* buffer, int height, int width, int channel, cv::Mat& mat);

// for common data
inline bool formatAsMatrix(const float* data, unsigned int rows, unsigned int cols, cv::Mat& mat);
inline bool formatAsMatrix(const int* data, unsigned int rows, unsigned int cols, cv::Mat& mat);

void splitImage(const cv::Mat& source, cv::Mat& image, cv::Mat& alpha);
void fuseImage(const cv::Mat& foreground, const cv::Mat& background, const cv::Mat& mask_f, cv::Mat& fusion);

#endif