
#include <cmath>
#include <cassert>
#include "xelement.h"



XPoint::XPoint()
	: x(0), y(0)
{

}

XPoint::XPoint(int xx, int yy)
	: x(xx), y(yy)
{

}


/*
 * class XRectangle implement
 */
XRectangle::XRectangle()
	: x_min(0), y_min(0), x_max(0), y_max(0)
{

}

void XRectangle::from(const int* points, int num_points)
{
	x_min = points[0], x_max = points[0];
	y_min = points[1], y_max = points[1];
	for (int n = 1; n < num_points; n++)
	{
		auto xx = points[n * 2 + 0];
		auto yy = points[n * 2 + 1];
		x_min = StdMin(xx, x_min);
		y_min = StdMin(yy, y_min);
		x_max = StdMax(xx, x_max);
		y_max = StdMax(yy, y_max);
	}
	assert(x_min < x_max && y_min < y_max);
}

void XRectangle::from(const float* points, int num_points)
{
	float xx_min = points[0], xx_max = points[0];
	float yy_min = points[1], yy_max = points[1];
	for (int n = 1; n < num_points; n++)
	{
		auto xx = points[n * 2 + 0];
		auto yy = points[n * 2 + 1];
		xx_min = StdMin(xx, x_min);
		yy_min = StdMin(yy, y_min);
		xx_max = StdMax(xx, x_max);
		yy_max = StdMax(yy, y_max);
	}
	x_min = static_cast<int>(round(xx_min));
	y_min = static_cast<int>(round(yy_min));
	x_max = static_cast<int>(round(xx_max));
	y_max = static_cast<int>(round(yy_max));
}

void XRectangle::from(const int* points, int num_points, const int* index, int num_index)
{
	assert(num_index <= num_points);
	x_min = points[index[0] * 2 + 0], x_max = points[index[0] * 2 + 0];
	y_min = points[index[0] * 2 + 1], y_max = points[index[0] * 2 + 1];
	for (int n = 1; n < num_index; n++)
	{
		int idx = index[n];
		auto xx = points[idx * 2 + 0];
		auto yy = points[idx * 2 + 1];
		x_min = StdMin(xx, x_min);
		y_min = StdMin(yy, y_min);
		x_max = StdMax(xx, x_max);
		y_max = StdMax(yy, y_max);
	}
	assert(x_min < x_max && y_min < y_max);
}

void XRectangle::expansion(float ratio_x, float ratio_y)
{
	assert(0 <= ratio_x && 0 <= ratio_y);
	int w = x_max - x_min + 1;
	int h = y_max - y_min + 1;
	x_min = static_cast<int>(x_min - w * ratio_x + 0.5);
	x_max = static_cast<int>(x_max + w * ratio_x + 0.5);
	y_min = static_cast<int>(y_min - h * ratio_y + 0.5);
	y_max = static_cast<int>(y_max + h * ratio_y + 0.5);
}

void XRectangle::expansion(float ratio_x_min, float ratio_x_max, float ratio_y_min, float ratio_y_max)
{
	int w = x_max - x_min + 1;
	int h = y_max - y_min + 1;
	x_min = static_cast<int>(x_min - w * ratio_x_min + 0.5);
	x_max = static_cast<int>(x_max + w * ratio_x_max + 0.5);
	y_min = static_cast<int>(y_min - h * ratio_y_min + 0.5);
	y_max = static_cast<int>(y_max + h * ratio_y_max + 0.5);
}

void XRectangle::clip(int xx_min, int yy_min, int xx_max, int yy_max)
{
	x_min = StdMax(xx_min, x_min);
	y_min = StdMax(yy_min, y_min);
	x_max = StdMin(xx_max, x_max);
	y_max = StdMin(yy_max, y_max);
}

void XRectangle::toSquare(bool expand)
{
	float h = height();
	float w = width();
	if (expand == false) 
	{
		float center_x = (x_min + x_max) / 2.0f;
		float center_y = (y_min + y_max) / 2.0f;
		float s = (w + h) / 2.0f;
		x_min = center_x - s / 2.0f;
		x_max = center_x + s / 2.0f;
		y_min = center_y - s / 2.0f;
		y_max = center_y + s / 2.0f;
	}
	else 
	{
		if (h > w) 
		{
			float sub = h - w;
			x_min -= sub / 2.0f;
			x_max += sub / 2.0f;
		}
		if (h < w) 
		{
			float sub = w - h;
			y_min -= sub / 2.0f;
			y_max += sub / 2.0f;
		}
	}
}

int XRectangle::height()
{
	return y_max - y_min + 1;
}

int XRectangle::width()
{
	return x_max - x_min + 1;
}

int XRectangle::area()
{
	return (y_max - y_min) * (x_max - x_min);
}

float XRectangle::iou(XRectangle& a, XRectangle& b)
{
	int xx1 = StdMax(a.x_min, b.x_min);
	int yy1 = StdMax(a.y_min, b.y_min);
	int xx2 = StdMin(a.x_max, b.x_max);
	int yy2 = StdMin(a.y_max, b.y_max);
	int inter_area = (StdMax(0, xx2 - xx1) * StdMax(0, yy2 - yy1));
	int area_a = a.area(), area_b = b.area();
	if (area_a == 0 || area_b == 0) return 0.f;
	float union_area = area_a + area_b - inter_area;
	return (union_area > 0.f) ? float(inter_area / union_area) : 0.f;
}

bool XRectangle::isValid(const int* points)
{
	return (points[0] < points[2]) && (points[1] < points[3]);
}