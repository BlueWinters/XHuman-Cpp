
#ifndef __visFunc__
#define __visFunc__

#include <opencv2\core\mat.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include "face_base/face_detection.h"


void draw4Corners(cv::Mat& mat, cv::Scalar& color, int thickness, int lft, int top, int rig, int bot);
void visBoxes(cv::Mat& mat, const int* box);
void visLandmarks(cv::Mat& mat, const int* points);
void visLandmarks(cv::Mat& mat, const int* points, int num_points, bool id);
void visIdentity(cv::Mat& mat, int id, const int* box);
void visDetection(cv::Mat& mat, const FaceObjectVector& vec);

void visText(cv::Mat& mat, const string str, int x, int y);


#endif 