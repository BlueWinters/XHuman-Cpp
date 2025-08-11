
#include <cassert>
#include <algorithm>
#include "face_detection.h"
#include "face_detection.mem.h"
#include "face_detection.id.h"
#include "xgeometry.h"

#ifndef StdMax
#define StdMax(a,b)  (((a) > (b)) ? (a) : (b))
#endif
#ifndef StdMin
#define StdMin(a,b)  (((a) < (b)) ? (a) : (b))
#endif


FaceDetector::FaceDetector()
{
	configure(0.5f, 0.4f);
	initialize();
}

FaceDetector::~FaceDetector()
{
	
}

void FaceDetector::doPadding(ncnn::Mat& mat, ResizeInfo& rsz_info)
{
	ncnn::Option opt(net.opt);
	Layer* layer = ncnn::create_layer("Padding");

	// set param
	ncnn::ParamDict pd;
	pd.set(0, rsz_info.top_pad);
	pd.set(1, rsz_info.bot_pad);
	pd.set(2, rsz_info.lft_pad);
	pd.set(3, rsz_info.rig_pad);
	pd.set(5, 255.f);

	ncnn::Mat tmp = mat.clone();
	layer->load_param(pd);
	layer->create_pipeline(opt);
	layer->forward(tmp, mat, opt);
	layer->destroy_pipeline(opt);
	delete layer;
}

void FaceDetector::decodeBox(ResizeInfo& rsz_info, const float* prior, const float* in_box, int* out_box)
{
	const float variance[2] = { 0.1, 0.2 };
	float cx = prior[0] + in_box[0] * variance[0] * prior[2];
	float cy = prior[1] + in_box[1] * variance[0] * prior[3];
	float w = prior[2] * exp(in_box[2] * variance[1]);
	float h = prior[3] * exp(in_box[3] * variance[1]);

	// x,y,w,h --> t,l,b,r
	float top = (cy - h / 2) * rsz_info.dst_h;  // y_min(t)
	float lft = (cx - w / 2) * rsz_info.dst_w;  // x_min(l)
	float bot = (cy + h / 2) * rsz_info.dst_h;  // y_max(b)
	float rig = (cx + w / 2) * rsz_info.dst_w;  // x_max(r)

	// map back to original coordinate
	float rescale_w = static_cast<float>(rsz_info.dst_w - rsz_info.lft_pad - rsz_info.rig_pad) / rsz_info.src_w;
	float rescale_h = static_cast<float>(rsz_info.dst_h - rsz_info.top_pad - rsz_info.bot_pad) / rsz_info.src_h;
	top = (top - rsz_info.top_pad) / rescale_h + rsz_info.top;
	lft = (lft - rsz_info.lft_pad) / rescale_w + rsz_info.lft;
	bot = (bot - rsz_info.top_pad) / rescale_h + rsz_info.top;
	rig = (rig - rsz_info.lft_pad) / rescale_w + rsz_info.lft;

	// transform to square box
	if (cfg_square_box == true)
	{
		const float src_box_bot = bot;
		float cx = (lft + rig) / 2.f;
		float cy = (top + bot) / 2.f;
		float s_w = (rig - lft);
		float s_h = (bot - top);
		float s4 = (s_w + s_h) / 4.f;
		// reshape to square
		lft = cx - s4; rig = cx + s4;
		top = cy - s4; bot = cy + s4;
		float diff = src_box_bot - bot;
		// extend box
		float ss = 2 * s4 * cfg_square_radio;
		lft -= ss;
		rig += ss;
		top = top + diff - ss;
		bot = bot + diff + ss;
	}

	// clip coordinate
	top = StdMax(StdMin(top, rsz_info.img_h - 1), 0);
	lft = StdMax(StdMin(lft, rsz_info.img_w - 1), 0);
	bot = StdMax(StdMin(bot, rsz_info.img_h - 1), 0);
	rig = StdMax(StdMin(rig, rsz_info.img_w - 1), 0);

	// assign & float --> int
	out_box[0] = static_cast<int>(round(lft));
	out_box[1] = static_cast<int>(round(top));
	out_box[2] = static_cast<int>(round(rig));
	out_box[3] = static_cast<int>(round(bot));
}

void FaceDetector::decodePoints(ResizeInfo& rsz_info, const float* ptr_prior, const float* in_points, int* out_points)
{
	const float variance = 0.1;
	const float xx = variance * ptr_prior[2];
	const float yy = variance * ptr_prior[3];
	float tmp_points[10];
	for (auto n = 0; n < 10; n += 2)
	{
		tmp_points[n + 0] = (ptr_prior[0] + in_points[n + 0] * xx) * rsz_info.dst_w;
		tmp_points[n + 1] = (ptr_prior[1] + in_points[n + 1] * yy) * rsz_info.dst_h;
	}

	// map back to original coordinate
	float rescale_w = static_cast<float>(rsz_info.dst_w - rsz_info.lft_pad - rsz_info.rig_pad) / rsz_info.src_w;
	float rescale_h = static_cast<float>(rsz_info.dst_h - rsz_info.top_pad - rsz_info.bot_pad) / rsz_info.src_h;
	for (auto n = 0; n < 10; n += 2)
	{
		tmp_points[n + 0] = (tmp_points[n + 0] - rsz_info.lft_pad) / rescale_w + rsz_info.top;
		tmp_points[n + 1] = (tmp_points[n + 1] - rsz_info.top_pad) / rescale_h + rsz_info.lft;
	}

	for (auto n = 0; n < 10; n+=2)
	{
		float x = StdMax(StdMin(tmp_points[n + 0], rsz_info.img_w - 1), 0);
		float y = StdMax(StdMin(tmp_points[n + 1], rsz_info.img_h - 1), 0);
		out_points[n+0] = static_cast<int>(round(x));
		out_points[n+1] = static_cast<int>(round(y));
	}
}

void FaceDetector::doNonMaxSuppression(FaceObjectVector& proposals, FaceObjectVector& obj_vec)
{
	if (proposals.empty())
		return;

	// sort by score
	auto sort_function = [](FaceObject* x, FaceObject* y){return x->score > y->score;};
	std::sort(proposals.begin(), proposals.end(), sort_function);

	// calculate each areas
	const int n = proposals.size();
	std::vector<int> picked;
	std::vector<float> areas(n);
	for (int i = 0; i < n; i++)
	{
		FaceObject& obj = *proposals[i];
		areas[i] = (obj.box[2] - obj.box[0]) * (obj.box[3] - obj.box[1]);
	}

	// calculate IoU
	for (int i = 0; i < n; i++)
	{
		int keep = 1;
		FaceObject& a = *proposals[i];
		for (int j = 0; j < (int)picked.size(); j++)
		{
			FaceObject& b = *proposals[picked[j]];

			// intersection over union
			int xx1 = StdMax(a.box[0], b.box[0]);
			int yy1 = StdMax(a.box[1], b.box[1]);
			int xx2 = StdMin(a.box[2], b.box[2]);
			int yy2 = StdMin(a.box[3], b.box[3]);
			float inter_area = static_cast<float>(StdMax(0, xx2 - xx1) * StdMax(0, yy2 - yy1));
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			if (inter_area / union_area > cfg_nms_threshold)
				keep = 0;
		}

		if (keep)
			picked.push_back(i);
	}

	// assign
	int picked_count = picked.size();
	for (int i = 0; i < picked_count; i++)
	{
		// TODO: do not clear the vector, just push back
		int index = picked[i];
		obj_vec.push_back(proposals[index]);
		//clean_proposals.push_back(proposals[index]);
		proposals[index] = NULL;
	}

	freeVector(proposals);
}

void FaceDetector::inference(ncnn::Mat& mat, ncnn::Mat& scores, ncnn::Mat& boxes, ncnn::Mat& points)
{
	//const float mean_val[3] = { 104, 117, 123 };
	//const float std_val[3] = { 1., 1., 1. };
	float mean_val[3] = { 0, 0, 0 };
	float std_val[3] = { 0.00392156f, 0.00392156f, 0.00392156f };
	mat.substract_mean_normalize(mean_val, std_val);

	ncnn::Extractor ex = net.create_extractor();
	//ex.set_num_threads(num_threads);
	ex.set_light_mode(light_mode);
	ex.input(FaceDetection_ParamID::BLOB_input, mat);
	ex.extract(FaceDetection_ParamID::BLOB_scores, scores);
	ex.extract(FaceDetection_ParamID::BLOB_boxes, boxes);
	ex.extract(FaceDetection_ParamID::BLOB_points, points);
}

void FaceDetector::preprocess(const unsigned char* data, int img_h, int img_w, int img_c,
	int dst_h, int dst_w, int y_min, int x_min, int y_max, int x_max, ncnn::Mat& mat, ResizeInfo& rsz_info)
{
	assert(x_min < x_max && y_min < y_max);
	assert(dst_w % 32 == 0 && dst_h % 32 == 0);

	y_min = StdMax(y_min, 0);
	y_max = StdMin(y_max, img_h - 1);
	x_min = StdMax(x_min, 0);
	x_max = StdMin(x_max, img_w - 1);

	int src_h = y_max - y_min + 1;
	int src_w = x_max - x_min + 1;
	float src_ratio = static_cast<float>(src_h) / src_w;
	float dst_ratio = static_cast<float>(dst_h) / dst_w;

	rsz_info.img_h = img_h;
	rsz_info.img_w = img_w;
	rsz_info.src_h = src_h;
	rsz_info.src_w = src_w;
	rsz_info.dst_h = dst_h;
	rsz_info.dst_w = dst_w;
	rsz_info.lft = x_min;
	rsz_info.rig = x_max;
	rsz_info.top = y_min;
	rsz_info.bot = y_max;
	if (src_ratio > dst_ratio)
	{
		rsz_info.rsz_h = dst_h;
		rsz_info.rsz_w = static_cast<int>(float(src_w) / src_h * dst_h);
		rsz_info.lft_pad = (dst_w - rsz_info.rsz_w) / 2;
		rsz_info.rig_pad = dst_w - rsz_info.rsz_w - rsz_info.lft_pad;
		rsz_info.top_pad = 0;
		rsz_info.bot_pad = 0;
	}
	else
	{
		rsz_info.rsz_h = static_cast<int>(float(src_h) / src_w * dst_w);;
		rsz_info.rsz_w = dst_w;
		rsz_info.top_pad = (dst_h - rsz_info.rsz_h) / 2;
		rsz_info.bot_pad = dst_h - rsz_info.rsz_h - rsz_info.top_pad;
		rsz_info.lft_pad = 0;
		rsz_info.rig_pad = 0;
	}

	auto type = (img_c == 1) ? ncnn::Mat::PIXEL_GRAY2BGR : ncnn::Mat::PIXEL_BGR;
	mat = ncnn::Mat::from_pixels_roi_resize(data, type, img_w, img_h, 
		rsz_info.lft, rsz_info.top, src_w, src_h, rsz_info.rsz_w, rsz_info.rsz_h);
	doPadding(mat, rsz_info);
}

void FaceDetector::postprocess(ncnn::Mat& scores, ncnn::Mat& boxes, ncnn::Mat& points, ResizeInfo& rsz_info, FaceObjectVector& vec)
{
	prior_box.config(rsz_info.dst_h, rsz_info.dst_w);

	const int num_proposals = scores.h;
	const float* ptr_score = scores.channel(0);
	const float* ptr_box = boxes.channel(0);
	const float* ptr_points = points.channel(0);
	const float* ptr_anchors = prior_box.ptr_anchors;

	// get proposals
	FaceObjectVector proposals;
	for (int n = 0; n < num_proposals; n++)
	{
		const float prob = ptr_score[1];
		if (prob >= cfg_score_threshold)
		{
			FaceObject* obj = new FaceObject();
			// 1.score
			obj->score = prob;
			// 2.box
			decodeBox(rsz_info, ptr_anchors, ptr_box, obj->box);
			// 3.points
			decodePoints(rsz_info, ptr_anchors, ptr_points, obj->points);
			// assign
			proposals.push_back(obj);
		}
		
		ptr_score += 2;
		ptr_box += 4;
		ptr_points += 10;
		ptr_anchors += 4;
	}

	// NMS
	doNonMaxSuppression(proposals, vec);
}

void FaceDetector::pipelineSingleScale(const unsigned char* data, int img_h, int img_w, int img_c,
	int dst_h, int dst_w, int y_min, int x_min, int y_max, int x_max, FaceObjectVector& obj_vec)
{
	ncnn::Mat image, scores, boxes, points;
	ResizeInfo rsz_info;
	preprocess(data, img_h, img_w, img_c, dst_h, dst_w, 
		y_min, x_min, y_max, x_max, image, rsz_info);
	inference(image, scores, boxes, points);
	postprocess(scores, boxes, points, rsz_info, obj_vec);
}

void FaceDetector::initialize()
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
	net.load_param(FaceDetection_Param_Bin);
	net.load_model(FaceDetection_Bin);
}

void FaceDetector::initialize(const char* path_param, const char* path_bin)
{
	assert(net.load_param(path_param) == 0 && net.load_model(path_bin) == 0);
}

void FaceDetector::configure(float score_threshold = 0.5, float nms_threshold = 0.4)
{
	cfg_score_threshold = score_threshold;
	cfg_nms_threshold = nms_threshold;
}

void FaceDetector::setSquareBoxes(bool enable)
{
	cfg_square_box = enable;
}

void FaceDetector::transformBox(const FaceObject& obj, XRectangle& rect)
{
	int points[8];
	FaceDetector::transformBox(obj, points);

	//for (auto n = 0; n < 4; n++)
	//{
	//	rect[n].x = points[2 * n + 0];
	//	rect[n].y = points[2 * n + 1];
	//}
}

void FaceDetector::transformBox(const FaceObject& obj, int* points)
{
	// calculate point and angle in Euclid space
	// that is, image coordinate --> world coordinate
	float le_x = obj.points[0];
	float le_y = -obj.points[1];
	float re_x = obj.points[2];
	float re_y = -obj.points[3];
	float ns_x = obj.points[4];
	float ns_y = -obj.points[5];

	double ie_x, ie_y;
	calculateIntersection(le_x, le_y, re_x, re_y, ns_x, ns_y, ie_x, ie_y);
	
	float vx = ie_x - ns_x;
	float vy = ie_y - ns_y;
	float angle = calculateAngle(vx, vy);
	float degree = angle / PI * 180;
	float rotate_angle = (vx > 0) ? (angle - PI / 2.f) : -(PI / 2.f - angle);

	int lft = obj.box[0];
	int top = -obj.box[1];
	int rig = obj.box[2];
	int bot = -obj.box[3];
	int box_points[8] = {lft, top, rig, top, rig, bot, lft, bot};

	for (auto n = 0; n < 4; n++)
	{
		double tx = box_points[2 * n + 0];
		double ty = box_points[2 * n + 1];
		rotatePoint(ns_x, ns_y, rotate_angle, tx, ty);
		points[2 * n + 0] = tx;
		points[2 * n + 1] = -ty;
	}
}

void FaceDetector::freeVector(FaceObjectVector& obj_vec)
{
	for (FaceObject* obj : obj_vec)
		if (obj != NULL) delete obj;
	obj_vec.clear();
}

void FaceDetector::detectSingleScale(const unsigned char* data, int in_height, int in_width, int in_channel, FaceObjectVector& obj_vec)
{
	pipelineSingleScale(data, in_height, in_width, in_channel, 
		DefaultHeight, DefaultWidth, 0, 0, in_height-1, in_width-1, obj_vec);
}

void FaceDetector::detectMultiScale(const unsigned char* data, int in_height, int in_width, int in_channel, FaceObjectVector& obj_vec)
{
	int max_size = StdMax(in_height, in_width);
	int rsz_h = DefaultHeight, rsz_w = DefaultWidth;

	// inference for every scale
	FaceObjectVector proposals;
	for ( ; rsz_h/2 < max_size || rsz_w/2 < max_size; rsz_h *= 2, rsz_w *= 2)
	{
		pipelineSingleScale(data, in_height, in_width, in_channel, rsz_h, rsz_w, 
			0, 0, in_height - 1, in_width - 1, proposals);
	}

	// NMS for all scale
	doNonMaxSuppression(proposals, obj_vec);
}

void FaceDetector::detectSpecific(const unsigned char* data, int in_height, int in_width, int in_channel,
	int y_min, int x_min, int y_max, int x_max, int rsz_height, int rsz_width, FaceObjectVector& obj_vec)
{
	pipelineSingleScale(data, in_height, in_width, in_channel, rsz_height, rsz_width, y_min, x_min, y_max, x_max, obj_vec);
}

