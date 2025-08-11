
#include <cassert>
#include <iostream>
#include "face_tracking.h"
#include "face_detection.h"
//#define XDebug_FaceTracking
#ifdef XDebug_FaceTracking
#include "tools/strfunc.h"
#endif

#ifndef round2Int
#define round2int(v)  static_cast<int>(round(v));
#endif


FaceTracking::FaceTracking()
	: face_detector(FaceDetector::getInstance())
	, face_align(FaceAlign::getInstance())
{
	assert(frequency_enter < sample_frequency);
}

FaceTracking::~FaceTracking()
{
	
}

void FaceTracking::initialize()
{
	face_detector.initialize();
	face_align.initialize();
	
}

void FaceTracking::setMode(FaceTrackingMode mode)
{
	this->mode = mode;
}

void FaceTracking::setFrequencyEnter(int frequency_enter)
{
	this->frequency_enter = frequency_enter;
}

void FaceTracking::finetuneFromLandmarks(FaceObject& obj)
{
	XRectangle rect;
	rect.from(obj.landmarks, FaceAlignNumPoints);
	rect.expansion(0.05f, 0.05f);

	int* box = obj.box;
	box[0] = rect.x_min;
	box[1] = rect.y_min;
	box[2] = rect.x_max;
	box[3] = rect.y_max;

	int* points = obj.points;
	int* landmarks = obj.landmarks;
	points[0] = (landmarks[36 * 2 + 0] + landmarks[39 * 2 + 0]) / 2;
	points[1] = (landmarks[36 * 2 + 1] + landmarks[39 * 2 + 1]) / 2;
	points[2] = (landmarks[42 * 2 + 0] + landmarks[45 * 2 + 0]) / 2;
	points[3] = (landmarks[42 * 2 + 1] + landmarks[45 * 2 + 1]) / 2;
	points[4] = (landmarks[30 * 2 + 0] + landmarks[33 * 2 + 0]) / 2;
	points[5] = (landmarks[30 * 2 + 1] + landmarks[33 * 2 + 1]) / 2;
	points[6] = landmarks[48 * 2 + 0];
	points[7] = landmarks[48 * 2 + 1];
	points[8] = landmarks[54 * 2 + 0];
	points[9] = landmarks[54 * 2 + 1];
}

void FaceTracking::smoothBox(const int* previous, int* current)
{
	for (int n = 0; n < 4; n++)
	{
		current[n] = round2int(current[n] * smooth_momentum + previous[n] * (1 - smooth_momentum));
	}
}


void FaceTracking::smoothLandmarks(const int* previous, int* current)
{
	for (int n = 0; n < FaceAlignNumPoints; n++)
	{
		current[n * 2 + 0] = round2int(current[n * 2 + 0] * smooth_momentum + previous[n * 2 + 0] * (1 - smooth_momentum));
		current[n * 2 + 1] = round2int(current[n * 2 + 1] * smooth_momentum + previous[n * 2 + 1] * (1 - smooth_momentum));
	}
}

void FaceTracking::sortObjects(FaceObjectVector& obj_vec)
{
	auto sort_function = (method == SortingMethod::Score) ? FaceTracking::sortByScore : FaceTracking::sortByArea;
	std::sort(obj_vec.begin(), obj_vec.end(), sort_function);
}

bool FaceTracking::sortByArea(const FaceObject* x, const FaceObject* y)
{
	if (x->identity != -1 && y->identity != -1)
	{
		if (x->frequency == y->frequency)
		{
			XRectangle xr, yr;
			xr.from(x->box, 2);
			yr.from(y->box, 2);
			return xr.area() > yr.area();
		}
		else return x->score > y->score;
	}
	else
	{
		return (x->identity > y->identity);
	}
}

bool FaceTracking::sortByScore(const FaceObject* x, const FaceObject* y)
{
	if (x->identity != -1 && y->identity != -1)
	{
		if (x->frequency == y->frequency)
			return x->score > y->score;
		else return x->frequency > y->frequency;
	}
	else
	{
		return (x->identity > y->identity);
	}
}

float FaceTracking::calculateIOU(const int* previous, const int* current, int num_points)
{
	XRectangle pre, cur;
	pre.from(previous, num_points);
	cur.from(current, num_points);
	float iou = XRectangle::iou(pre, cur);
	return iou;
}

float FaceTracking::calculateArea(const int* points, int num_points)
{
	XRectangle rect;
	rect.from(points, num_points);
	return rect.area();
}

float FaceTracking::calculateFlow(const int* previous, const int* current, int num_points)
{
	int flow_x = abs(previous[0] - current[0]);
	int flow_y = abs(previous[1] - current[1]);
	float flow_max = sqrt(flow_x ^ 2 + flow_y ^ 2);
	for (int n = 1; n < num_points; n++)
	{
		flow_x = abs(previous[0] - current[0]);
		flow_y = abs(previous[1] - current[1]);
		float flow = sqrt(flow_x ^ 2 + flow_y ^ 2);
		flow_max = StdMax(flow, flow_max);
		previous += 2;
		current += 2;
	}
	return flow_max;
}

bool FaceTracking::checkValid(const int* previous, const int* current, int num_points, 
	float& iou, float& area, float& flow)
{
	assert(num_points == FaceAlignNumPoints);
	iou = calculateIOU(previous, current, num_points);
	area = calculateArea(previous, num_points);
	flow = calculateFlow(previous, current, num_points) / static_cast<float>(area);
	return bool((iou > iou_threshold || flow < points_flow_maximum) && area > face_area_minimum);
}

void FaceTracking::detectLocal(const unsigned char* input, int in_height, int in_width,
	int in_channel, const FaceObject& obj, FaceObjectVector& obj_vec)
{
	XRectangle rect;
	rect.from(obj.landmarks, FaceAlignNumPoints);
	rect.expansion(1.f, 1.f);
	face_detector.detectSpecific(input, in_height, in_width, in_channel,
		rect.y_min, rect.x_min, rect.y_max, rect.x_max, local_crop_height, local_crop_width, obj_vec);
}

void FaceTracking::smoothPosition(const FaceObject& pre, FaceObject& cur)
{
	cur.score = (pre.score + cur.score) / 2.f;
	cur.frequency = pre.frequency;
	if (cur.hit() > frequency_enter)
	{
		// satisfy condition for assigning id
		if (pre.identity == FaceIdentityInvalid)
			cur.identity = ++total_number;
		else cur.identity = pre.identity;
	}

	smoothBox(pre.box, cur.box);
	smoothLandmarks(pre.landmarks, cur.landmarks);
	finetuneFromLandmarks(cur);
}

void FaceTracking::smoothPosition(const int* landmarks, FaceObject& cur)
{
	if (cur.hit() > frequency_enter)
	{
		// satisfy condition for assigning id, then calculate landmarks
		if (cur.identity == FaceIdentityInvalid)
			cur.identity = ++total_number;
	}

	smoothLandmarks(landmarks, cur.landmarks);
	finetuneFromLandmarks(cur);
}

int FaceTracking::findBestMatch(const FaceObjectVector& obj_vec, FaceObject& cur)
{
	// find the best match in previous
	int idx_max = -1; 
	float iou_max = 0.f;
	for (int n = 0; n < obj_vec.size(); n++)
	{
		float iou = 0.f;
		const FaceObject& obj = *obj_vec[n];
		if (obj.identity != FaceIdentityInvalid)
			iou = calculateIOU(obj.landmarks, cur.landmarks, FaceAlignNumPoints);
		else iou = calculateIOU(obj.points, cur.points, 5);
		if (iou > iou_max)
		{
			iou_max = iou;
			idx_max = n;
		}
	}
	return idx_max;
}

void FaceTracking::updateCommon(const unsigned char* input, int in_height, int in_width, 
	int in_channel, unsigned int frame_num, FaceObjectVector& obj_vec)
{
	int num_pre = obj_vec.size();

	FaceObjectVector obj_vec_tmp;
	face_detector.detectSingleScale(input, in_height, in_width, in_channel, obj_vec_tmp);

	// assign objects
	FaceObjectVector obj_vec_cur;
	for (int n = 0; n < obj_vec_tmp.size(); n++)
	{
		FaceObject& obj_cur = *obj_vec_tmp[n];

		// calculate landmarks
		face_align.pipeline(input, in_height, in_width, in_channel,
			2, obj_cur.box, obj_cur.landmarks);
		
		// find the best match in previous
		int index = findBestMatch(obj_vec, obj_cur);

		// match at least one object, then merge the previous into current
		if (index != -1)
		{
			const FaceObject& obj_pre_match = *obj_vec[index];
			smoothPosition(obj_pre_match, obj_cur);
		#ifdef XDebug_FaceTracking
			string line = formatString("common: match & erase, %d %d", 
				obj_pre_match.identity, obj_pre_match.frequency);
			std::cout << line << endl;
		#endif
			obj_vec_cur.push_back(obj_vec_tmp[n]);
			obj_vec_tmp[n] = NULL;
		}
		else
		{
			// new object
			if (frequency_enter == 0)
				obj_cur.identity = ++total_number;
			obj_vec_cur.push_back(obj_vec_tmp[n]);
			obj_vec_tmp[n] = NULL;
		}
	}

	// replace
	obj_vec.swap(obj_vec_cur);
	FaceDetector::freeVector(obj_vec_cur);
	FaceDetector::freeVector(obj_vec_tmp);

	// sort
	sortObjects(obj_vec);
}

void FaceTracking::updateFastAllFaces(const unsigned char* input, int in_height, int in_width, 
	int in_channel, unsigned int frame_num, FaceObjectVector& obj_vec)
{
	if (frame_num % sample_frequency != 0)
	{
		// just tracking
		if (obj_vec.size() > 0)
		{
			FaceObjectVector obj_vec_tmp;
			for (int n = 0; n < obj_vec.size(); n++)
			{
				FaceObject& obj = *obj_vec[n];

				// re-detect based on landmarks
				int landmarks[FaceAlignNumPoints * 2];
				memcpy(landmarks, obj.landmarks, sizeof(int) * FaceAlignNumPoints * 2);
				face_align.pipeline(input, in_height, in_width, in_channel,
					FaceAlignNumPoints, landmarks, obj.landmarks);

				// check based on IOU & area & flow
				float iou, area, flow;
				if (checkValid(landmarks, obj.landmarks,
					FaceAlignNumPoints, iou, area, flow) == true)
				{
					smoothPosition(landmarks, obj);
					obj_vec_tmp.push_back(obj_vec[n]);
					obj_vec[n] = NULL;
				}
				else
				{
					delete obj_vec[n];
					obj_vec[n] = NULL;
				}
			}

			FaceDetector::freeVector(obj_vec);
			obj_vec.swap(obj_vec_tmp);
		}
		else
		{
			// no object in previous frame, re-detecting
			updateCommon(input, in_height, in_width, in_channel, frame_num, obj_vec);
		}
	}
	else
	{
		// re-detecting
		updateCommon(input, in_height, in_width, in_channel, frame_num, obj_vec);
	}

	sortObjects(obj_vec);
}

void FaceTracking::updateFastOneFace(const unsigned char* input, int in_height, int in_width,
	int in_channel, unsigned int frame_num, FaceObjectVector& obj_vec)
{
	// when sample_frequency is 1, it gives automatic verification
	if (frame_num % sample_frequency != 0)
	{
		// keep tracking
		if (obj_vec.size() > 0)
		{
			FaceObject& obj = *obj_vec[0];

			// get landmarks
			int landmarks[FaceAlignNumPoints * 2];
			memcpy(landmarks, obj.landmarks, sizeof(int) * FaceAlignNumPoints * 2);
			face_align.pipeline(input, in_height, in_width, in_channel,
				FaceAlignNumPoints, landmarks, obj.landmarks);

			// check based on IOU
			float iou, area, flow;
			if (checkValid(landmarks, obj.landmarks,
				FaceAlignNumPoints, iou, area, flow) == true)
			{
				// smooth position in data(box,points)
				smoothPosition(landmarks, obj);
			}
			else
			{
			#ifdef XDebug_FaceTracking
				std::cout << formatString("fastOne: remove %.2f, %.2f, %.2f", iou, area, flow) << endl;
			#endif
				// remove all objects
				FaceDetector::freeVector(obj_vec);
			}
		}
		else
		{
			updateCommon(input, in_height, in_width, in_channel, frame_num, obj_vec);
		}
	}
	else
	{
		if (obj_vec.size() > 0)
		{
			// remain the first
			FaceObject& obj = *obj_vec[0];

			// detecting in local
			FaceObjectVector obj_vec_tmp;
			detectLocal(input, in_height, in_width, in_channel, obj, obj_vec_tmp);

			int size = obj_vec_tmp.size();
			if (size > 0)
			{
				int index = (size == 1) ? 0 : findBestMatch(obj_vec_tmp, obj);
				if (index != -1)
				{
					FaceObject* cur = obj_vec_tmp[index];
					face_align.pipeline(input, in_height, in_width, in_channel,
						FaceAlignNumPoints, obj.landmarks, cur->landmarks);
					smoothPosition(obj, *cur);
					obj_vec_tmp[index] = NULL;
					FaceDetector::freeVector(obj_vec);
					FaceDetector::freeVector(obj_vec_tmp);
					obj_vec.push_back(cur);
				}
				else
				{
					// no match objects
					FaceDetector::freeVector(obj_vec);
					FaceDetector::freeVector(obj_vec_tmp);
				}
			}
			else
			{
				// no objects in local, detecting global
				updateCommon(input, in_height, in_width, in_channel, frame_num, obj_vec);
			}
		}
		else
		{
			// no object in previous frame, re-detecting
			updateCommon(input, in_height, in_width, in_channel, frame_num, obj_vec);
		}
	}

	// sort by area
	sortObjects(obj_vec);

	// retain only one face
	if (obj_vec.size() > 1)
	{
		for (int n = 1; n < obj_vec.size(); n++)
			delete obj_vec[n];
		obj_vec.erase(++obj_vec.begin(), obj_vec.end());
	}
}

void FaceTracking::pipelineUpdate(const unsigned char* input, 
	int in_height, int in_width, int in_channel, unsigned int frame_num, FaceObjectVector& obj_vec)
{
	if (frame_num > 0)
	{
		if (mode == FaceTrackingMode::Common)
			updateCommon(input, in_height, in_width, in_channel, frame_num, obj_vec);
		else if (mode == FaceTrackingMode::FastAllFace)
			updateFastAllFaces(input, in_height, in_width, in_channel, frame_num, obj_vec);
		else if (mode == FaceTrackingMode::FastOneFace)
			updateFastOneFace(input, in_height, in_width, in_channel, frame_num, obj_vec);
	}
	else
	{
		// frame index start from 0
		updateCommon(input, in_height, in_width, in_channel, frame_num, obj_vec);
	}
}
