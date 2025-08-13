
#include <iostream>
#include "tools/ximage.h"
#include "tools/timer.h"
#include "tools/strfunc.h"
#include "tools/visfunc.h"
#include "face_3dmm/face_3dmm.h"
#include "face_3dmm/face_render.h"


void main_FaceMasking()
{
	Face3DMM face_3dmm;
	face_3dmm.initialize("face_reconstruction.ncnn.param", "face_reconstruction.ncnn.bin");
	FaceRender face_render;
	face_render.initialize("face_masking.bin");

	cv::VideoCapture capture(0);
	assert(capture.isOpened());
	int h = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	int w = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	cout << formatString("open camera: (%d, %d)", h, w) << endl;

	cv::Mat mat;
	FaceObjectVector info_vector;
	unsigned int counter = 0;
	cv::Mat uv_texture = cv::imread("texture.png", cv::IMREAD_UNCHANGED);

	// update
	bool flag_is_texture = 0;
	bool flag_flip = false;
	int fps = 0;
	float sum = 0, cost = 0, mean = 0;
	while (capture.read(mat) == true)
	{
		if (flag_flip == true) cv::flip(mat, mat, 1);
		XImage image(mat);
		// modeling
		auto beg = getTimeInUs();
		Face3DMMResultVector result_vector;
		face_3dmm.inference(image, result_vector);
		FaceRenderResult result_render;
		if (flag_is_texture)
			face_render.inference(result_vector[0], uv_texture, result_render);
		else face_render.inference(result_vector[0], result_render);
		FaceRenderResult result_source;
		face_render.pasteBack(result_vector[0], result_render, image.cv_mat, result_source);
		auto end = getTimeInUs();
		// time & fps
		sum += cost = (end - beg) / 1000.f;
		mean = sum / ++counter;
		fps = 1000.f / mean;
		// visual
		visText(result_source.image, formatString("mean: %2dms", static_cast<int>(mean + 0.5)), w * 0.8, h * 0.05);
		visText(result_source.image, formatString("delay: %2dms", static_cast<int>(cost + 0.5)), w * 0.8, h * 0.1);
		visText(result_source.image, formatString("fps: %2d", static_cast<int>(fps)), w * 0.8, h * 0.15);
		cv::imshow("show", result_source.image);
		int key = cv::waitKey(1);
		if (key == 'q') break;
		if (key == 'f') flag_flip = !flag_flip;
		if (key == ' ') flag_is_texture = !flag_is_texture;
	}

	FaceDetector::freeVector(info_vector);
	capture.release();
}

#if 1
int main(int argc, char** argv)
{
	main_FaceMasking();

#ifdef _MSC_VER
	system("pause");
#endif
}
#endif