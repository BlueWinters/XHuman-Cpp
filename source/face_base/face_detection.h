
#ifndef __Face_Detection__
#define __Face_Detection__

#include "ncnn/net.h"
#include "priorbox.h"
#include "face_info.h"
#include "xelement.h"
#include "singleton.h"

using namespace std;
using namespace ncnn;


class FaceDetector
{
public:
	THREAD_SAFE_SINGLETON_AUTOMATIC(FaceDetector);
public:
	FaceDetector();
	virtual ~FaceDetector();

protected:
	int num_threads = 2;
	bool light_mode = false;
	bool use_gpu = true;
	ncnn::Net net;
protected:
	PriorBox prior_box;
	float cfg_nms_threshold = 0.3f;
	float cfg_score_threshold = 0.5f;
	int cfg_topk_keep = 5000;
	bool cfg_square_box = false;
	float cfg_square_radio = 0.02f;

protected:
	class ResizeInfo
	{
	public:
		int img_h, img_w;
		int src_h, src_w;
		int rsz_h, rsz_w;
		int dst_h, dst_w;
		int top, bot;
		int lft, rig;
		int top_pad, bot_pad;
		int lft_pad, rig_pad;
	};

protected:
	void doPadding(ncnn::Mat& mat, ResizeInfo& rsz_info);
	void decodeBox(ResizeInfo& rsz_info, const float* prior, const float* in_box, int* out_box);
	void decodePoints(ResizeInfo& rsz_info, const float* ptr_prior, const float* in_points, int* out_points);
	void doNonMaxSuppression(FaceObjectVector& proposals, FaceObjectVector& obj_vec);
	void preprocess(const unsigned char* data, int img_h, int img_w, int img_c, int dst_h, int dst_w,
		int y_min, int x_min, int y_max, int x_max, ncnn::Mat& mat, ResizeInfo& rsz_info);
	void inference(ncnn::Mat& mat, ncnn::Mat& scores, ncnn::Mat& boxes, ncnn::Mat& points);
	void postprocess(ncnn::Mat& boxes, ncnn::Mat& scores, ncnn::Mat& points, ResizeInfo& rsz_info, FaceObjectVector& vec);
	void pipelineSingleScale(const unsigned char* data, int img_h, int img_w, int img_c,
		int dst_h, int dst_w, int y_min, int x_min, int y_max, int x_max, FaceObjectVector& obj_vec);

public:
	virtual void initialize();
	void initialize(const char* path_param, const char* path_bin);
	void configure(float score_threshold, float nms_threshold);
	void setSquareBoxes(bool enable);
	void detectSingleScale(const unsigned char* data, int in_height, int in_width, int in_channel, FaceObjectVector& obj_vec);
	void detectMultiScale(const unsigned char* data, int in_height, int in_width, int in_channel, FaceObjectVector& obj_vec);
	void detectSpecific(const unsigned char* data, int in_height, int in_width, int in_channel,
		int y_min, int x_min, int y_max, int x_max, int rsz_height, int rsz_width, FaceObjectVector& obj_vec);

public:
	static void transformBox(const FaceObject& obj, XRectangle& rect);
	static void transformBox(const FaceObject& obj, int* points);
	static void freeVector(FaceObjectVector& obj_vec);

public:
	static const int DefaultHeight = 640;
	static const int DefaultWidth = 640;
};


#endif
