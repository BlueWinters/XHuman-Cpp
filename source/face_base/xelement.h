
#ifndef __XElement__
#define __XElement__


#ifndef StdMax
#define StdMax(a,b)  (((a) > (b)) ? (a) : (b))
#endif
#ifndef StdMin
#define StdMin(a,b)  (((a) < (b)) ? (a) : (b))
#endif



class XPoint
{
public:
	XPoint();
	XPoint(int xx, int yy);

public:
	int x, y;
	inline int operator[] (size_t i)
	{
		int* p = (int*)this;
		return p[i];
	};
};


class XRectangle
{
public:
	XRectangle();

public:
	int x_min, y_min;
	int x_max, y_max;

public:
	void from(const int* points, int num_points);
	void from(const int* points, int num_points, const int* index, int num_index);
	void from(const float* points, int num_points);
	void expansion(float ratio_x, float ratio_y);
	void expansion(float ratio_x_min, float ratio_x_max, float ratio_y_min, float ratio_y_max);
	void clip(int xx_min, int yy_min, int xx_max, int yy_max);
	void toSquare(bool expand = false);
public:
	int height();
	int width();
	int area();
public:
	static float iou(XRectangle& a, XRectangle& b);
	static bool isValid(const int* points);
};


#endif