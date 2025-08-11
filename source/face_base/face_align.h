
#ifndef __Face_Align__
#define __Face_Align__

#include "ncnn/net.h"
#include "face_detection.h"
#include "xelement.h"
#include "singleton.h"

using namespace ncnn;


class FaceAlign
{
public:
	THREAD_SAFE_SINGLETON_AUTOMATIC(FaceAlign);
protected:
	FaceAlign();
	virtual ~FaceAlign();

protected:
	int num_threads = 2;
	bool light_mode = false;
	bool use_gpu = true;
	ncnn::Net net;

protected:
	void calculateBoundingBox(const int* previous, int num_points, XRectangle& rect);
	void preprocess(const unsigned char* input, int in_height, int in_width, int in_channel,
		const int* previous, int num_points, ncnn::Mat& mat, XRectangle& rect);
	void inference(ncnn::Mat& input, ncnn::Mat& output); 
	void postprocess(int in_height, int in_width, XRectangle& rect, ncnn::Mat& mat, int* output);

public:
	virtual void initialize();
	void initialize(const char* path_param, const char* path_bin);
	void pipeline(const unsigned char* input, int in_height, int in_width, 
		int in_channel, int num_points, const int* points, int* landmarks);

public:
	//static const int FaceAlignNumPoints = 68;
	static const int FaceAlignNormHeight = 128;
	static const int FaceAlignNormWidth = 128;
public:
	// index of points of brows, eyes, nose, mouth and face contour
	static const int IndexBrowLft[5];
	static const int IndexBrowRig[5];
	static const int IndexEyeLft[6];
	static const int IndexEyeRig[6];
	static const int IndexNose[9];
	static const int IndexMouth[12];
	static const int IndexContour[17];
};

#endif
