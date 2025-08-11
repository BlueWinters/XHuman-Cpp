
#ifndef __XGeometry__
#define __XGeometry__

#include <cmath>
#include <cfloat>
#include <cassert>

#ifndef PI
#define PI  3.141592653589793
#endif

#ifndef isZero
#define isZero(v) (::fabs(v) < DBL_EPSILON)
#endif

inline void rotatePoint(double &x, double &y, double angle)
{
	auto cos_angle = ::cos(angle);
	auto sin_angle = ::sin(angle);
	auto tmp_x = x;
	auto tmp_y = y;
	x = cos_angle * tmp_x - sin_angle * tmp_y;
	y = sin_angle * tmp_x + cos_angle * tmp_y;
}

inline void rotatePoint(double ox, double oy, double angle, double& x, double& y)
{
	auto tx = x - ox;
	auto ty = y - oy;
	rotatePoint(tx, ty, angle);
	x = tx + ox;
	y = ty + oy;
}

inline double calculateAngle(double vx, double vy)
{
	double l = sqrt(vx * vx + vy * vy);
	double angle = ::acos(vx / l);
	return (vy > 0) ? angle : (2 * PI - angle);
}

inline bool calculateIntersection(double sx, double sy, double ex, double ey, 
	double px, double py, double& ix, double& iy)
{
	// coefficient of linear equation: ax+by+c=0
	double a = sy - ey;
	double b = ex - sx;
	double c = -(a * sx + b * sy);

	double vx = ey - sy;
	double vy = -(ex - sx);
	double ppx = px - vx;
	double ppy = py - vy;

	double aa = py - ppy;
	double bb = ppx - px;
	double cc = -(aa * px + bb * py);

	double base = a * bb - b * aa;

	if (isZero(base) == true)
		return false;

#ifdef _DEBUG
	// just for debug
	double dix = (b * cc - c * bb) / base;
	double diy = (c * aa - a * cc) / base;
	double v = (py - diy) * (sy - ey) + (px - dix) * (sx - ex);
	assert(isZero(v));
#endif

	ix = (b * cc - c * bb) / base;
	iy = (c * aa - a * cc) / base;
	return true;
}

#endif
