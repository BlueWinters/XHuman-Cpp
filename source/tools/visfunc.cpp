
#include "visfunc.h"
#include "strfunc.h"
#include "cvfunc.h"


void draw4Corners(cv::Mat& mat, cv::Scalar& color, int thickness, int lft, int top, int rig, int bot)
{
	int h = bot - top;
	int w = rig - lft;
	int size = h * 0.1;
	//
	cv::line(mat, cv::Point(lft, top), cv::Point(lft + size, top), color, thickness);
	cv::line(mat, cv::Point(lft, top), cv::Point(lft, top + size), color, thickness);
	//
	cv::line(mat, cv::Point(rig, top), cv::Point(rig - size, top), color, thickness);
	cv::line(mat, cv::Point(rig, top), cv::Point(rig, top + size), color, thickness);
	//
	cv::line(mat, cv::Point(rig, bot), cv::Point(rig - size, bot), color, thickness);
	cv::line(mat, cv::Point(rig, bot), cv::Point(rig, bot - size), color, thickness);
	//
	cv::line(mat, cv::Point(lft, bot), cv::Point(lft + size, bot), color, thickness);
	cv::line(mat, cv::Point(lft, bot), cv::Point(lft, bot - size), color, thickness);
}

void visBoxes(cv::Mat& mat, const int* box)
{
	int lft = box[0];
	int top = box[1];
	int rig = box[2];
	int bot = box[3];
	int thickness = 3;
	cv::Point lft_top(lft, top);
	cv::Point rig_bot(rig, bot);
	auto color = cv::Scalar(0, 255, 0);  // green
	draw4Corners(mat, color, thickness, lft, top, rig, bot);
	//drawBox(mat, lft, top, rig, top, lft, bot, rig, bot);
	//drawRetangle(mat, lft, top, rig, bot);
}

void visLandmarks(cv::Mat& mat, const int* points)
{
	cv::circle(mat, cv::Point(points[0], points[1]), 1, cv::Scalar(0, 0, 225), 4);   // red
	cv::circle(mat, cv::Point(points[2], points[3]), 1, cv::Scalar(0, 255, 225), 4); // yellow
	cv::circle(mat, cv::Point(points[4], points[5]), 1, cv::Scalar(255, 0, 225), 4); // magenta
	cv::circle(mat, cv::Point(points[6], points[7]), 1, cv::Scalar(0, 255, 0), 4);   // green
	cv::circle(mat, cv::Point(points[8], points[9]), 1, cv::Scalar(255, 0, 0), 4);   // blue
}

void visLandmarks(cv::Mat& mat, const int* points, int num_points, bool id)
{
	int radius = 1;
	double font_scale = 0.5;
	int thickness = 2;
	int font_face = cv::FONT_HERSHEY_COMPLEX;

	for (int n = 0; n < num_points; n++)
	{
		int x = points[2 * n + 0], y = points[2 * n + 1];
		stringstream sbuffer;
		sbuffer << n;
		cv::circle(mat, cv::Point(x, y), radius, (cv::Scalar(255, 0, 0)));
		if (id == true)
		{
			cv::putText(mat, cv::String(sbuffer.str()), cv::Point(x, y),
				font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 2, 0);
		}
	}
}

void visIdentity(cv::Mat& mat, int id, const int* box)
{
	double font_scale = 0.5;
	int thickness = 2;
	int font_face = cv::FONT_HERSHEY_COMPLEX;

	int lft = box[0], top = box[1];
	cv::putText(mat, formatString("%d", id).c_str(), cv::Point(lft, top - 5),
		font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 2, 0);
}

void visDetection(cv::Mat& mat, const FaceObjectVector& vec)
{
	for (auto n = 0; n < vec.size(); n++)
	{
		FaceObject& obj = *vec[n];
		//visBoxes(mat, obj.box);
		//visLandmarks(mat, obj.points);
		visLandmarks(mat, obj.landmarks, 68, false);
		//visIdentity(mat, obj.identity, obj.box);
	}
}

void visText(cv::Mat& mat, const string str, int x, int y)
{
	double font_scale = 0.5;
	int font_face = cv::FONT_HERSHEY_SIMPLEX;
	int thickness = 1;
	cv::putText(mat, cv::String(str.c_str()), cv::Point(x, y),
		font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 2, 0);
}