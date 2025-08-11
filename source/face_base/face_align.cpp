
#include <cassert>
#include <iostream>
#include "face_align.h"
#include "face_align.id.h"
#include "face_align.mem.h"


#ifndef StdMax
#define StdMax(a,b)  (((a) > (b)) ? (a) : (b))
#endif
#ifndef StdMin
#define StdMin(a,b)  (((a) < (b)) ? (a) : (b))
#endif


const int FaceAlign::IndexBrowLft[5] = { 17, 18, 19, 20, 21 };
const int FaceAlign::IndexBrowRig[5] = { 22, 23, 24, 25, 26 };
const int FaceAlign::IndexEyeLft[6] = { 36, 37, 38, 39, 40, 41 };
const int FaceAlign::IndexEyeRig[6] = { 42, 43, 44, 45, 46, 47 };
const int FaceAlign::IndexNose[9] = { 27, 28, 29, 30, 31, 32, 33, 34, 35 };
const int FaceAlign::IndexMouth[12] = { 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 };
const int FaceAlign::IndexContour[17] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };


FaceAlign::FaceAlign()
{
	initialize();
}

FaceAlign::~FaceAlign()
{

}

void FaceAlign::initialize()
{
#if NCNN_VULKAN
	// enable vulkan compute feature before loading
	net.opt.use_vulkan_compute = use_gpu;
	net.opt.num_threads = num_threads;
	net.opt.use_fp16_packed = false;
	net.opt.use_fp16_storage = false;
	net.opt.use_fp16_arithmetic = false;
	net.opt.use_int8_storage = false;
	net.opt.use_int8_arithmetic = false;
#endif
	net.load_param(FaceAlign_OptParamBin);
	net.load_model(FaceAlign_OptBin);
}

void FaceAlign::initialize(const char* path_param, const char* path_bin)
{
	assert(net.load_param(path_param) == 0 && net.load_model(path_bin) == 0);
}

void FaceAlign::calculateBoundingBox(const int* previous, int num_points, XRectangle& rect)
{
	rect.from(previous, num_points);
	if (num_points == 2)
	{
		// TODO: nothing to do for 2 points (top left and right bottom)
		;
	}
	if (num_points == FaceAlignNumPoints)
	{
		float ratio_x_min = -0.05f, ratio_x_max = +0.05f;
		float ratio_y_min = +0.05f, ratio_y_max = +0.05f;
		rect.expansion(ratio_x_min, ratio_x_max, ratio_y_min, ratio_y_max);
	}
	if (num_points == 5)
	{
		float ratio_x = 0.5f, ratio_y = 0.5f;
		rect.expansion(ratio_x, ratio_y);
	}
}

void FaceAlign::preprocess(const unsigned char* input,
	int in_height, int in_width, int in_channel, const int* previous, int num_points, ncnn::Mat& mat, XRectangle& rect)
{
	// calculate bounding box
	calculateBoundingBox(previous, num_points, rect);
	//rect.clip(0, 0, in_width - 1, in_height - 1);

	// clip the face from the original face 
    int clip_width = rect.width();
	int clip_height = rect.height();
	unsigned char* clip_buffer = new unsigned char[clip_height * clip_width * in_channel]();
	for (int c = 0; c < in_channel; c++) {
		for (int h = 0; h < clip_height; h++) 
			for (int w = 0; w < clip_width; w++) {		
				int src_w = w + rect.x_min;
				int src_h = h + rect.y_min;
				if (0 <= src_w && src_w < in_width && 0 <= src_h && src_h < in_height) {
					clip_buffer[in_channel * (h * clip_width + w) + c] =
						input[in_channel * (src_h * in_width + src_w) + c];
			}
		}
	}


	// convert to ncnn::Mat
	const float mean_vals[] = { 0.f, 0.f, 0.f };
	const float norm_vals[] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
	mat = ncnn::Mat::from_pixels_resize(clip_buffer, ncnn::Mat::PIXEL_BGR, clip_width, clip_height,
		FaceAlignNormWidth, FaceAlignNormHeight);
	mat.substract_mean_normalize(mean_vals, norm_vals);

	// release buffer
	delete[] clip_buffer;
}

void FaceAlign::inference(ncnn::Mat& input, ncnn::Mat& output)
{
	ncnn::Extractor ex = net.create_extractor();
	ex.set_light_mode(light_mode);
	ex.input(FaceAlign_OptParamID::BLOB_input, input);
	ex.extract(FaceAlign_OptParamID::BLOB_output, output);
}

void FaceAlign::postprocess(int in_height, int in_width, XRectangle& rect, ncnn::Mat& mat, int* output)
{
	int clip_width = rect.width();
	int clip_height = rect.height();
	const float* data = mat.channel(0).row(0);
	for (int n = 0; n < FaceAlignNumPoints; n++)
	{
		output[n * 2 + 0] = static_cast<int>(round(data[n * 2 + 0] * clip_width + rect.x_min));
		output[n * 2 + 1] = static_cast<int>(round(data[n * 2 + 1] * clip_height + rect.y_min));
	}
}

void FaceAlign::pipeline(const unsigned char* input, int in_height, int in_width, 
	int in_channel, int num_points, const int* points, int* landmarks)
{
	XRectangle rect;
	ncnn::Mat mat_input, mat_output;
	assert(num_points == 2 || num_points == FaceAlignNumPoints);
	preprocess(input, in_height, in_width, in_channel, points, num_points, mat_input, rect);
	inference(mat_input, mat_output);
	postprocess(in_height, in_width, rect, mat_output, landmarks);
}
