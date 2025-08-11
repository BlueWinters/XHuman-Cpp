
#include <opencv2/opencv.hpp>
#include "face_3dmm.h"
#include "face_base/face_detection.h"
#include "face_base/face_align.h"


#ifndef StdMax
#define StdMax(a,b)  (((a) > (b)) ? (a) : (b))
#endif
#ifndef StdMin
#define StdMin(a,b)  (((a) < (b)) ? (a) : (b))
#endif

const float Landmark3D[] = {
	-0.311487f,
	+0.290361f,
	+0.133780f,
	+0.309799f,
	+0.289720f,
	+0.131795f,
	+0.003253f,
	-0.046179f,
	+0.552442f,
	-0.252169f,
	-0.381339f,
	+0.224057f,
	+0.248466f,
	-0.381282f,
	+0.222358f
};


Face3DMM::Face3DMM()
{
	face_tracker.setMode(FaceTracking::FaceTrackingMode::FastOneFace);
}

Face3DMM::~Face3DMM()
{
}

void Face3DMM::initialize()
{

}

void Face3DMM::initialize(const char* path_param, const char* path_bin)
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
	assert(net.load_param(path_param) == 0 && net.load_model(path_bin) == 0);
}

void Face3DMM::calculate5Points(const int* landmark, cv::Mat& mat, int height)
{
	int points[10];
	// left-eye
	points[0] = (landmark[2 * 36 + 0] + landmark[2 * 39 + 0]) / 2;
	points[1] = (landmark[2 * 36 + 1] + landmark[2 * 39 + 1]) / 2;
	// right-eye
	points[2] = (landmark[2 * 42 + 0] + landmark[2 * 45 + 0]) / 2;
	points[3] = (landmark[2 * 42 + 1] + landmark[2 * 45 + 1]) / 2;
	// nose
	points[4] = landmark[2 * 30 + 0];
	points[5] = landmark[2 * 30 + 1];
	// left-mouth-corner
	points[6] = landmark[2 * 48 + 0];
	points[7] = landmark[2 * 48 + 1];
	// right-mouth-corner
	points[8] = landmark[2 * 54 + 0];
	points[9] = landmark[2 * 54 + 1];

	// reformat
	mat = cv::Mat(5, 2, CV_32FC1);
	for (int i = 0; i < 5; i++)
	{
		mat.ptr<float>(i)[0] = points[i * 2 + 0];
		mat.ptr<float>(i)[1] = height - 1 - points[i * 2 + 1];
	}
}

void Face3DMM::calculateParameters(const cv::Mat& xp, const float* x, float* t, float& s)
{
	const int npts = xp.size[0];
	cv::Mat A = cv::Mat::zeros(2 * npts, 8, CV_32FC1);
	for (int i = 0; i < npts; i++)
	{
		A.ptr<float>(i * 2)[0] = x[i * 3];
		A.ptr<float>(i * 2)[1] = x[i * 3 + 1];
		A.ptr<float>(i * 2)[2] = x[i * 3 + 2];
		A.ptr<float>(i * 2)[3] = 1;

		A.ptr<float>(i * 2 + 1)[4] = x[i * 3];
		A.ptr<float>(i * 2 + 1)[5] = x[i * 3 + 1];
		A.ptr<float>(i * 2 + 1)[6] = x[i * 3 + 2];
		A.ptr<float>(i * 2 + 1)[7] = 1;
	}
	vector<int> newshape = { 2 * npts, 1 };
	cv::Mat b = xp.reshape(0, newshape);
	cv::Mat k;
	cv::solve(A.t() * A, A.t() * b, k, cv::DECOMP_CHOLESKY);
	const float* pk = (float*)k.data;
	float norm_R1 = sqrt(pk[0] * pk[0] + pk[1] * pk[1] + pk[2] * pk[2]);
	t[0] = pk[3];
	float norm_R2 = sqrt(pk[4] * pk[4] + pk[5] * pk[5] + pk[6] * pk[6]);
	t[1] = pk[7];
	s = (norm_R1 + norm_R2) * 0.5f;
}

void Face3DMM::cropImage(const cv::Mat& image_bgr, cv::Mat& image_cropped, const float* t, const float s, const int target_size, FormatInfo& format_info)
{
	const float h0 = image_bgr.rows;
	const float w0 = image_bgr.cols;
	const int w = int(w0 * s);
	const int h = int(h0 * s);
	int lft = int(w * 0.5f - target_size * 0.5f + (t[0] - w0 * 0.5f) * s);  // 避免使用除法
	int rig = std::min(int(lft + target_size), w);    // 防止超出边界
	int top = int(h * 0.5f - target_size * 0.5f + (h0 * 0.5f - t[1]) * s);
	int bot = std::min(int(top + target_size), h);   // 防止超出边界

	cv::Mat image_resized;
	cv::resize(image_bgr, image_resized, cv::Size(w, h), cv::INTER_CUBIC);
	lft = StdMax(lft, 0);
	rig = StdMin(rig, image_resized.cols);
	top = StdMax(top, 0);
	bot = StdMin(bot, image_resized.rows);
	format_info.h = h;
	format_info.w = w;
	format_info.lft = lft;
	format_info.rig = rig;
	format_info.top = top;
	format_info.bot = bot;
	format_info.pad_h = 0;
	format_info.pad_w = 0;
	image_resized(cv::Rect(lft, top, rig - lft, bot - top)).copyTo(image_cropped);
	const int dst_h = target_size;
	const int dst_w = target_size;
	if (image_resized.rows < dst_h)
	{
		// padding to bottom
		format_info.pad_h = dst_h - image_resized.rows;
		cv::copyMakeBorder(image_resized, image_resized, 0, dst_h - image_resized.rows, 0, 0, BORDER_CONSTANT, 0);
	}
	if (image_resized.cols < dst_w)
	{
		// padding to right
		format_info.pad_w = dst_w - image_resized.cols;
		cv::copyMakeBorder(image_resized, image_resized, 0, 0, 0, dst_w - image_resized.cols, BORDER_CONSTANT, 0);
	}
}

void Face3DMM::formatInput(const cv::Mat& image, const int* landmarks, cv::Mat& image_cropped, FormatInfo& format_info)
{
	int height = image.rows;
	// estimate align parameters
	cv::Mat points;
	calculate5Points(landmarks, points, height);
	// crop image
	float t[2], s;
	calculateParameters(points, Landmark3D, t, s);
	cropImage(image, image_cropped, t, this->rescale_factor / s, this->target_size, format_info);
	// copy landmarks
	std::memcpy(format_info.landmark, landmarks, sizeof(format_info.landmark));
}

void Face3DMM::forward(cv::Mat& image_cropped, Face3DMMResult& result)
{
	ncnn::Mat input, output;
	XImage ximage_cropped(image_cropped);
	// normalize
	const float mean_vals[] = { 0.f, 0.f, 0.f };
	const float norm_vals[] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
	input = ncnn::Mat::from_pixels(ximage_cropped.data, ncnn::Mat::PIXEL_BGR2RGB, ximage_cropped.height, ximage_cropped.width);
	input.substract_mean_normalize(mean_vals, norm_vals);
	// forward
	ncnn::Extractor ex = net.create_extractor();
	ex.set_light_mode(true);
	ex.input("in0", input);
	ex.extract("out0", output);
	// postprocess
	const float* data = output.channel(0).row(0);
	std::memcpy(&result.coefficients, data, sizeof(result.coefficients));
	//for (int n = 0; n < 257; n++) {
	//	printf("%+.6f\n", result.coefficients.parameters[n]);
	//}
}

void Face3DMM::inference(XImage& image, Face3DMMResultVector& result_vector)
{
	FaceDetector& face_detector = FaceDetector::getInstance();
	FaceAlign& face_align = FaceAlign::getInstance();
	FaceObjectVector object_vector;
	face_detector.detectSingleScale(image.data, image.height, image.width, image.channel, object_vector);

	const int num_objects = object_vector.size();
	result_vector.resize(num_objects);
	for (int n = 0; n < num_objects; n++)
	{
		FaceObject* object = object_vector[n];
		Face3DMMResult& result = result_vector[n];
		// estimate 68-points
		face_align.pipeline(image.data, image.height, image.width, image.channel, 2, object->box, object->landmarks);
		// format input
		cv::Mat image_cropped;
		formatInput(image.cv_mat, object->landmarks, image_cropped, result.format_info);
		forward(image_cropped, result);
	}

	FaceDetector::freeVector(object_vector);
}
