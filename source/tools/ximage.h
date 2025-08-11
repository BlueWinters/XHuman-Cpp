
#ifndef __XImage__
#define __XImage__

#include "cvfunc.h"
#include <opencv2\core\mat.hpp>

#define ModeUnknown			0
#define ModePixelWise		1
#define ModeChannelWise		2


class XImage
{
public:
	const unsigned char* data = nullptr;
	int height = 0;
	int width = 0;
	int channel = 0;
	int mode = ModeUnknown;
	cv::Mat cv_mat;

public:
	// 默认构造函数
	XImage();
	// 构造函数
	XImage(int height, int width, int channel, const unsigned char* image, int mode);
	// opencv的mat构造函数
	explicit XImage(const cv::Mat& mat);
	// 拷贝构造函数
	XImage(const XImage& other);
	// 赋值操作符
	XImage& operator = (const XImage& other);
	// 移动构造函数
	XImage& operator = (XImage&& other) noexcept;
	// 析构函数
	virtual ~XImage();

public:
	// 从其他图像拷贝数据
	void from(const XImage& other);
	// 获取图像的分辨率（高x宽）
	inline size_t size() const;
	// 清理图像
	inline void clean();
	// 获取像素值
	inline unsigned char at(int x, int y, int c) const;
	// 是否是空图像
	inline bool empty() const;
};



#endif