
#include <iostream>
#include "tools/ximage.h"
#include "tools/timer.h"
#include "tools/strfunc.h"
#include "tools/visfunc.h"
#include "face_base/face_tracking.h"
#include "face_3dmm/face_3dmm.h"
#include "face_3dmm/face_render.h"


void testImage_FaceDetection()
{
	FaceDetector face_detector;
	cv::Mat image_bgr = cv::imread("D:\\Project\\cython-extension\\face_mesh_render\\asset\\obama_source.png");
	XImage image(image_bgr);
	FaceObjectVector info_vector;
	face_detector.detectSingleScale(image.data, image.height, image.width, image.channel, info_vector);
	visDetection(image_bgr, info_vector);
	cv::imshow("show", image_bgr);
	cv::waitKey(0);
}

void testImage_Face3DMM()
{
	Face3DMM face_3dmm;
	face_3dmm.initialize(
		"X:/checkpoints/deep-3d-reconstruction/pnnx/face_reconstruction.ncnn.param",
		"X:/checkpoints/deep-3d-reconstruction/pnnx/face_reconstruction.ncnn.bin"
	);
	FaceRender face_render;

	cv::Mat image_bgr = cv::imread("D:\\Project\\cython-extension\\face_mesh_render\\asset\\obama_source.png");
	cv::Mat uv_texture = cv::imread("D:\\Project\\cython-extension\\face_mesh_render\\asset\\texture.png", cv::IMREAD_UNCHANGED);
	XImage image(image_bgr);
	
	Face3DMMResultVector result_vector;
	face_3dmm.inference(image, result_vector);

	FaceRenderResult result_render;
	face_render.inference(result_vector[0], uv_texture, result_render);

	FaceRenderResult result_source;
	face_render.pasteBack(result_vector[0], result_render, image_bgr, result_source);

	cv::imwrite("N:\\archive\\2025\\0807-3dmm\\image.png", result_source.image);
	cv::imwrite("N:\\archive\\2025\\0807-3dmm\\depth.png", result_source.depth);
	cv::imwrite("N:\\archive\\2025\\0807-3dmm\\mask.png", result_source.mask);
	//cv::imshow("show", result_render.image);
	//cv::waitKey(0);
}

void testCamera_FaceTracking()
{
	FaceTracking face_tracking;
	face_tracking.setMode(FaceTracking::FaceTrackingMode::FastOneFace);

	auto capture = cv::VideoCapture(0);
	//capture.open("N:/dataset/dance-video/dance-3.sub.mp4");
	int h = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	int w = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	cout << formatString("open camera: (%d, %d)", h, w) << endl;

	cv::Mat mat;
	FaceObjectVector info_vector;
	unsigned int counter = 0;

	// update
	float sum = 0, cost = 0, mean = 0;
	while (capture.read(mat) == true)
	{
		cv::flip(mat, mat, 1);
		XImage image(mat);
		// tracking
		auto beg = getTimeInUs();
		face_tracking.pipelineUpdate(image.data, image.height, image.width,
			image.channel, counter++, info_vector);
		auto end = getTimeInUs();
		// time & fps
		sum += cost = (end - beg) / 1000.f;
		mean = sum / ++counter;
		// visual
		visDetection(mat, info_vector);
		visText(mat, formatString("mean: %2dms", static_cast<int>(mean + 0.5)), w * 0.8, h * 0.05);
		visText(mat, formatString("delay: %2dms", static_cast<int>(cost + 0.5)), w * 0.8, h * 0.1);
		cv::imshow("show", mat);
		int key = cv::waitKey(1);
		if (key == 'q') break;
		switch (key)
		{
		case 'c': face_tracking.setMode(FaceTracking::FaceTrackingMode::Common); break;
		case 'a': face_tracking.setMode(FaceTracking::FaceTrackingMode::FastAllFace); break;
		case 'o': face_tracking.setMode(FaceTracking::FaceTrackingMode::FastOneFace); break;
		};
	}

	FaceDetector::freeVector(info_vector);
	capture.release();
}

void testCamera_FaceMasking()
{
	Face3DMM face_3dmm;
	face_3dmm.initialize(
		"X:/checkpoints/deep-3d-reconstruction/pnnx/face_reconstruction.ncnn.param",
		"X:/checkpoints/deep-3d-reconstruction/pnnx/face_reconstruction.ncnn.bin"
	);
	FaceRender face_render;
	face_render.initialize("X:\\checkpoints\\deep-3d-reconstruction\\cpp\\face_masking.bin");

	auto capture = cv::VideoCapture(0);
	int h = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	int w = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	cout << formatString("open camera: (%d, %d)", h, w) << endl;

	cv::Mat mat;
	FaceObjectVector info_vector;
	unsigned int counter = 0;

	cv::Mat uv_texture = cv::imread("D:\\Project\\cython-extension\\face_mesh_render\\asset\\texture.png", cv::IMREAD_UNCHANGED);

	// update
	int fps = 0;
	float sum = 0, cost = 0, mean = 0;
	while (capture.read(mat) == true)
	{
		cv::flip(mat, mat, 1);
		XImage image(mat);
		// modeling
		auto beg = getTimeInUs();
		Face3DMMResultVector result_vector;
		face_3dmm.inference(image, result_vector);
		FaceRenderResult result_render;
		//face_render.inference(result_vector[0], uv_texture, result_render);
		face_render.inference(result_vector[0], result_render);
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
	}

	FaceDetector::freeVector(info_vector);
	capture.release();
}

#if 1
int main(int argc, char** argv)
{
	//testImage_FaceDetection();
	//testImage_Face3DMM();
	//testCamera_FaceTracking();
	testCamera_FaceMasking();

	//cv::Mat mat3(10, 10, CV_32FC3);
	//cv::Mat mat2(10, 10, CV_32FC1);
	//cv::Mat mat1(10, 10, CV_32F);

	//int tt1 = CV_32F;
	//int tt2 = CV_32FC1;
	//int tt3 = CV_32FC3;
	//int tt4 = CV_32FC4;

	//int c1 = mat1.channels();
	//int c2 = mat2.channels();

	//int t1 = mat1.type();
	//int t2 = mat2.type();
	//int t3 = mat3.type();

	//cv::Mat face_weighted;
	//cv::multiply(mat1, mat2, face_weighted);

#ifdef _MSC_VER
	system("pause");
#endif
}
#endif