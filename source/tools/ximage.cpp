
#include <memory>
#include <cstring>
#include <stdexcept>
#include "ximage.h"
#include "cvfunc.h"


// 默认构造函数
XImage::XImage()
{
    data = nullptr;
    height = 0;
    width = 0;
    channel = 0;
    mode = ModeUnknown;
}

// 构造函数
XImage::XImage(int height, int width, int channel, const unsigned char* image, int mode)
    : height(height), width(width), channel(channel), mode(mode)
{
    if (height <= 0 || width <= 0 || channel <= 0) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    if (mode != ModeChannelWise && mode != ModePixelWise) {
        throw std::invalid_argument("Invalid image pixel-format");
    }

    // 设置属性
    this->height = height;
    this->width = width;
    this->channel = channel;
    this->mode = mode;
    // 分配内存
    size_t buffer_size = static_cast<size_t>(height) * static_cast<size_t>(width) * static_cast<size_t>(channel);
    unsigned char* buffer = new unsigned char[buffer_size];
    if (buffer == nullptr) {
        throw std::bad_alloc();
    }
    // 初始化data
    std::memcpy(buffer, image, buffer_size * sizeof(unsigned char));
    this->data = buffer;
    // 初始化mat
    if (mode == ModeChannelWise) {
        formatBufferC2Mat(data, height, width, channel, this->cv_mat);
    }
    else {
        formatBufferP2Mat(data, height, width, channel, this->cv_mat);
    }
}

// OpenCV的Mat构造函数
XImage::XImage(const cv::Mat& mat)
{
    if (mat.empty()) {
        throw std::invalid_argument("Input Mat is empty");
    }

    // 初始化mat
    cv_mat = mat.clone();
    // 默认是Pixel-Wise格式
    unsigned char* buffer = nullptr;
    formatCVMat2BufferP(cv_mat, this->height, this->width, this->channel, &buffer);
    this->mode = ModePixelWise;
    this->data = buffer;
}

// 拷贝构造函数
XImage::XImage(const XImage& other)
{
    from(other);
}

// 赋值操作符
XImage& XImage::operator = (const XImage& other)
{
    if (this != &other) {
        clean();
        from(other);
    }
    return *this;
}

// 移动赋值操作符
XImage& XImage::operator=(XImage && other) noexcept
{
    if (this != &other) {
        // 释放当前资源
        clean();

        // 移动资源
        this->height = other.height;
        this->width = other.width;
        this->channel = other.channel;
        this->mode = other.mode;
        this->data = other.data;
        this->cv_mat = std::move(other.cv_mat);

        // 重置源对象
        other.height = 0;
        other.width = 0;
        other.channel = 0;
        other.mode = ModeUnknown;
        other.data = nullptr;
        other.cv_mat = cv::Mat();
    }
    return *this;
}

// 析构函数
XImage::~XImage()
{
    clean();
}

// 获取图像的分辨率（高x宽）
inline size_t XImage::size() const
{
    return static_cast<size_t>(height) * static_cast<size_t>(width);
}

// 是否是空图像
bool XImage::empty() const
{
    return (height == 0 || width == 0 || channel == 0 || data == nullptr);
}

// 清理图像
void XImage::clean()
{
    if (data != nullptr) {
        height = 0;
        width = 0;
        channel = 0;
        mode = ModeUnknown;
        delete[] data;
        data = nullptr;
        cv_mat = cv::Mat();
    }
}

void XImage::from(const XImage& other)
{
    if (other.empty() == true) {
        throw std::invalid_argument("Input is empty");
    }

    this->height = other.height;
    this->width = other.width;
    this->channel = other.channel;
    this->mode = other.mode;
    this->cv_mat = other.cv_mat.clone();
    size_t size = static_cast<size_t>(height) * static_cast<size_t>(width) * static_cast<size_t>(channel);
    unsigned char* buffer = new unsigned char[size];
    std::memcpy(buffer, other.data, size * sizeof(unsigned char));
    this->data = buffer;
}