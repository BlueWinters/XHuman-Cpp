
#ifndef __Face_Tracking__
#define __Face_Tracking__

#include "face_align.h"
#include "face_detection.h"


class FaceTracking
{
public:
	FaceTracking();
	~FaceTracking();

public:
	enum class FaceTrackingMode
	{
		Common = 1, FastOneFace = 2, FastAllFace = 3
	};
	enum class SortingMethod
	{
		Score = 1, Area = 2,
	};

protected:
	// base handle
	FaceDetector& face_detector;
	FaceAlign& face_align;
	// tracking config
	FaceTrackingMode mode = FaceTrackingMode::FastAllFace;
	SortingMethod method = SortingMethod::Score;
	int total_number = 0;
	bool auto_detect = true;
	int frequency_enter = 0;
	int sample_frequency = 4;
	float iou_threshold = 0.6f;
	int face_area_minimum = 32 * 32;
	float points_flow_maximum = 0.0625;
	float smooth_momentum = 0.8f;
	int local_crop_height = 160;
	int local_crop_width = 160;

protected:
	float calculateIOU(const int* previous, const int* current, int num_points);
	float calculateArea(const int* points, int num_points);
	float calculateFlow(const int* previous, const int* current, int num_points);
	bool checkValid(const int* previous, const int* current, int num_points, float& iou, float& area, float& flow);
	void detectLocal(const unsigned char* input, int in_height, int in_width,
		int in_channel, const FaceObject& obj, FaceObjectVector& obj_vec);
	void smoothPosition(const FaceObject& pre, FaceObject& cur);
	void smoothPosition(const int* landmarks, FaceObject& cur);
	int findBestMatch(const FaceObjectVector& obj_vec, FaceObject& cur);
	void finetuneFromLandmarks(FaceObject& obj);
	void smoothBox(const int* previous, int* current);
	void smoothLandmarks(const int* previous, int* current);
	void sortObjects(FaceObjectVector& obj_vec);
	static bool sortByArea(const FaceObject* x, const FaceObject* y);
	static bool sortByScore(const FaceObject* x, const FaceObject* y);

protected:
	void updateCommon(const unsigned char* input, int in_height, int in_width, 
		int in_channel, unsigned int frame_num, FaceObjectVector& obj_vec);
	void updateFastAllFaces(const unsigned char* input, int in_height, int in_width,
		int in_channel, unsigned int frame_num, FaceObjectVector& obj_vec);
	void updateFastOneFace(const unsigned char* input, int in_height, int in_width, 
		int in_channel, unsigned int frame_num, FaceObjectVector& obj_vec);

public:
	void initialize();
	void setMode(FaceTrackingMode mode);
	void setFrequencyEnter(int frequency_enter);
	void pipelineUpdate(const unsigned char* input, int in_height, int in_width, int in_channel, 
		unsigned int frame_num, FaceObjectVector& obj_vec);
};

#endif
