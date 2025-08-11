
#include <cmath>
#include <fstream>
#include "priorbox.h"


PriorBox::PriorBox()
	: clip(false)
	, steps{8, 16, 32, 64}
{
	height = width = 0;
	num_anchors = 0;
	ptr_anchors = NULL;
	
	//anchor_size_config.resize(4);
	anchor_size_config[0] = { 10, 20, 40 };
	anchor_size_config[1] = { 64, 96 };
	anchor_size_config[2] = { 128, 192 };
	anchor_size_config[3] = { 256, 384, 512 };
}

PriorBox::~PriorBox()
{
	clear();
}

//void PriorBox::config(int in_height, int in_width)
//{
//	if (in_height == height && in_width == width)
//		return; // do not need to config
//
//	// clear previous config
//	clear();
//
//	// config anchors according to input size
//	height = in_height;
//	width = in_width;
//	int feat_size_height[3];
//	int feat_size_width[3];
//	for (int n = 0; n < num_fpn; n++)
//	{
//		feat_size_height[n] = std::ceil(float(height) / steps[n]);
//		feat_size_width[n] = std::ceil(float(width) / steps[n]);
//	}
//
//	num_anchors = 0;
//	for (int n = 0; n < num_fpn; n++)
//		num_anchors += feat_size_height[n] * feat_size_width[n] * 2;
//
//	ptr_anchors = new float[num_anchors * 4];
//	float* header = ptr_anchors;
//	for (int n = 0; n < num_fpn; n++)
//	{
//		for (int i = 0; i < feat_size_height[n]; i++)
//		{
//			for (int j = 0; j < feat_size_width[n]; j++)
//			{
//				float s1_kx = float(min_size1[n]) / width;
//				float s1_ky = float(min_size1[n]) / height;
//				float s2_kx = float(min_size2[n]) / width;
//				float s2_ky = float(min_size2[n]) / height;
//				float dense_cx = (j + 0.5f) * steps[n] / width;
//				float dense_cy = (i + 0.5f) * steps[n] / height;
//				*(header++) = dense_cx;
//				*(header++) = dense_cy;
//				*(header++) = s1_kx;
//				*(header++) = s1_ky;
//				*(header++) = dense_cx;
//				*(header++) = dense_cy;
//				*(header++) = s2_kx;
//				*(header++) = s2_ky;
//			}
//		}
//	}
//}

void PriorBox::config(int in_height, int in_width)
{
	if (in_height == height && in_width == width)
		return; // do not need to config

	// clear previous config
	clear();

	// config anchors according to input size
	height = in_height;
	width = in_width;
	int feat_size_height[num_fpn];
	int feat_size_width[num_fpn];
	for (int n = 0; n < num_fpn; n++)
	{
		feat_size_height[n] = std::ceil(float(height) / steps[n]);
		feat_size_width[n] = std::ceil(float(width) / steps[n]);
	}

	// count anchors
	num_anchors = 0;
	for (int n = 0; n < num_fpn; n++)
		num_anchors += feat_size_height[n] * feat_size_width[n] * anchor_size_config[n].size();

	// assign
	ptr_anchors = new float[num_anchors * 4];
	float* header = ptr_anchors;
	for (int n = 0; n < num_fpn; ++n)
	{
		std::vector<int>& anchor_size = anchor_size_config[n];
		for (int i = 0; i < feat_size_height[n]; ++i)
		{
			for (int j = 0; j < feat_size_width[n]; ++j)
			{
				for (int k = 0; k < anchor_size.size(); ++k)
				{
					float s_kx = float(anchor_size[k]) / width;
					float s_ky = float(anchor_size[k]) / height;
					float cx = (j + 0.5) * steps[n] / width;
					float cy = (i + 0.5) * steps[n] / height;
					*(header++) = cx;
					*(header++) = cy;
					*(header++) = s_kx;
					*(header++) = s_ky;
				}
			}
		}
	}
}

void PriorBox::clear()
{
	if (ptr_anchors != NULL)
	{
		delete[] ptr_anchors;
		ptr_anchors = NULL;
		num_anchors = 0;
	}
}

void PriorBox::dump(const char* path)
{
#if defined (XDebug) && (_MSC_VER)
	if (ptr_anchors != NULL)
	{
		std::fstream file(path, 'w');
		if (file.is_open() == false)
			return;

		for (int n = 0; n < num_anchors; n++)
		{
			char buffer[64];
			sprintf_s(buffer, 64, "%+.6f,%+.6f,%+.6f,%+.6f,\n",
				ptr_anchors[n * 4 + 0], ptr_anchors[n * 4 + 1],
				ptr_anchors[n * 4 + 2], ptr_anchors[n * 4 + 3]);
			file << buffer;
		}

		file.close();
	}
#endif
}
