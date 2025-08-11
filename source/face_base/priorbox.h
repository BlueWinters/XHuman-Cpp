
#ifndef __PriorBox__
#define __PriorBox__

#include <vector>


class PriorBox
{
public:
	PriorBox();
	~PriorBox();

protected:
	static const int num_fpn = 4;
	const bool clip;
	const int steps[num_fpn];
	std::vector<int> anchor_size_config[num_fpn];
	
public:
	int height;
	int width;
	int num_anchors;
	float* ptr_anchors;

public:
	void config(int in_height, int in_width);
	void clear();
	void dump(const char* path);
};



#endif

