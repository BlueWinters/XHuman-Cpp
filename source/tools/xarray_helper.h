
#ifndef __XArray_Helper__
#define __XArray_Helper__

#include <opencv2\opencv.hpp>
#include "xarray.h"


// transform array to opencv matrix
void transformXArray2Matrix(const XArray& array, cv::Mat& mat);

// load from file
bool int32fromFile(const char* path, cv::Mat& mat);
bool float32fromFile(const char* path, cv::Mat& mat);

#endif
