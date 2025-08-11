
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>
#include <algorithm>
#include <omp.h>

#ifndef StdMax
#define StdMax(a,b)  (((a) > (b)) ? (a) : (b))
#endif
#ifndef StdMin
#define StdMin(a,b)  (((a) < (b)) ? (a) : (b))
#endif


bool formatCVMat2BufferC(const cv::Mat& mat, int& height, int& width, int& channel, unsigned char** buffer) 
{
    // 检查输入参数
    if (mat.empty() || buffer == nullptr) {
        return false;
    }

    // 获取图像信息
    height = mat.rows;
    width = mat.cols;
    channel = mat.channels();

    // 只支持1通道（灰度）和3通道（BGR）图像
    if (mat.type() != CV_8UC1 && mat.type() != CV_8UC3) {
        return false;
    }

    // 确保Mat数据是连续存储的
    const cv::Mat input = mat.isContinuous() ? mat : mat.clone();
    const unsigned char* data = input.data;
    size_t total_pixels = static_cast<size_t>(height) * static_cast<size_t>(width);

    // 计算需要的缓冲区大小
    size_t buffer_size = (channel == 3) ? 3 * total_pixels : total_pixels;

    // 分配内存
    *buffer = new unsigned char[buffer_size];
    unsigned char* ptr = *buffer;
    if (channel == 3) {
        // 3通道图像：BGR -> BBBBGGGGRRRR (Channel-Wise)
        // 使用OpenMP并行化
        #pragma omp parallel for num_threads(3) schedule(static)
        for (int c = 0; c < 3; ++c) {
            for (size_t i = 0; i < total_pixels; ++i) {
                ptr[c * total_pixels + i] = data[i * 3 + c];
            }
        }
    }
    else if (channel == 1) {
        // 1通道图像：直接复制
        std::memcpy(ptr, data, total_pixels * sizeof(unsigned char));
        //for (size_t i = 0; i < total_pixels; ++i) {
        //    ptr[i] = data[i * channel];
        //}
    }

    return true;
}


bool formatCVMat2BufferP(const cv::Mat& mat, int& height, int& width, int& channel, unsigned char** buffer) 
{
    // 检查输入参数
    if (mat.empty() || buffer == nullptr) {
        return false;
    }

    // 获取图像信息
    height = mat.rows;
    width = mat.cols;
    channel = mat.channels();

    // 只支持1通道（灰度）和3通道（BGR）图像
    if (mat.type() != CV_8UC1 && mat.type() != CV_8UC3) {
        return false;
    }

    // 确保Mat数据是连续存储的
    cv::Mat input = mat.isContinuous() ? mat : mat.clone();
    const unsigned char* data = input.data;
    size_t total_pixels = static_cast<size_t>(height) * static_cast<size_t>(width) * static_cast<size_t>(channel);

    // 分配内存
    *buffer = new unsigned char[total_pixels];

    // 直接复制数据（Pixel-Wise格式与OpenCV的BGR格式相同）
    std::memcpy(*buffer, data, total_pixels);
    return true;
}

bool formatBufferP2Mat(const unsigned char* buffer, int height, int width, int channel, cv::Mat& mat) 
{
    // 检查输入参数
    if (buffer == nullptr || height <= 0 || width <= 0 || channel <= 0) {
        return false;
    }

    // 只支持1通道（灰度）和3通道（BGR）图像
    if (channel != 1 && channel != 3) {
        return false;
    }

    size_t total_pixels = static_cast<size_t>(height) * static_cast<size_t>(width) * static_cast<size_t>(channel);
    mat.create(height, width, CV_8UC(channel));
    std::memcpy(mat.data, buffer, total_pixels);
    return !mat.empty();
}

bool formatBufferC2Mat(const unsigned char* buffer, int height, int width, int channel, cv::Mat& mat) 
{
    // 检查输入参数
    if (buffer == nullptr || height <= 0 || width <= 0 || channel <= 0) {
        return false;
    }

    // 只支持1通道（灰度）和3通道（BGR）图像
    if (channel != 1 && channel != 3) {
        return false;
    }

    size_t total_pixels = static_cast<size_t>(height) * static_cast<size_t>(width);

    if (channel == 3) {
        // 3通道图像：BBBBGGGGRRRR -> BGR (Channel-Wise to Pixel-Wise)
        mat = cv::Mat(height, width, CV_8UC3);
        unsigned char* data = mat.data;

        #pragma omp parallel for num_threads(4) schedule(static)
        for (int i = 0; i < total_pixels; ++i) {
            data[i * 3 + 0] = buffer[i];                     // B
            data[i * 3 + 1] = buffer[total_pixels + i];      // G
            data[i * 3 + 2] = buffer[2 * total_pixels + i];  // R
        }
    }
    else if (channel == 1) {
        // 1通道图像：直接创建
        mat.create(height, width, CV_8UC1);
        std::memcpy(mat.data, buffer, total_pixels);
    }

    return !mat.empty();
}

bool formatAsMatrix(const float* data, unsigned int rows, unsigned int cols, cv::Mat& mat) {
    // 检查输入参数
    if (data == nullptr || rows == 0 || cols == 0) {
        return false;
    }
    // 创建CV_32F类型的Mat（OpenCV中float32的类型）
    mat = cv::Mat(rows, cols, CV_32F);
    // 复制数据
    std::memcpy(mat.data, data, rows * cols * sizeof(float));
    return !mat.empty();
}

bool formatAsMatrix(const int* data, unsigned int rows, unsigned int cols, cv::Mat& mat) {
    // 检查输入参数
    if (data == nullptr || rows == 0 || cols == 0) {
        return false;
    }
    // 创建CV_32S类型的Mat（OpenCV中int32的类型）
    mat = cv::Mat(rows, cols, CV_32S);
    // 复制数据
    std::memcpy(mat.data, data, rows * cols * sizeof(float));
    return !mat.empty();
}

void splitImage(const cv::Mat& source, cv::Mat& image, cv::Mat& alpha)
{
    // 1. 检查输入是否为空
    if (source.empty()) {
        throw std::invalid_argument("Input source is empty");
    }

    // 2. 检查通道数是否为 4
    if (source.channels() != 4) {
        throw std::invalid_argument("Source must be a 4-channel matrix");
    }

    // 4. 获取基本参数
    const int rows = source.rows;
    const int cols = source.cols;
    const size_t total = rows * cols;

    // 5. 创建输出矩阵：image = H×W×3, alpha = H×W×1
    // 类型自动匹配 source
    if (source.type() == CV_8UC4) {
        image.create(rows, cols, CV_8UC3);
        alpha.create(rows, cols, CV_8UC1);
        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                auto& data_src = source.at<cv::Vec4b>(i, j);
                auto& data_img = image.at<cv::Vec3b>(i, j);
                data_img[0] = static_cast<uchar>(data_src[0]);
                data_img[1] = static_cast<uchar>(data_src[1]);
                data_img[2] = static_cast<uchar>(data_src[2]);
                alpha.at<uchar>(i, j) = static_cast<uchar>(data_src[3]);
            }
        }
    }
    else if (source.type() == CV_32FC4) {
        image.create(rows, cols, CV_32FC3);
        alpha.create(rows, cols, CV_32FC1);
        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                auto& data_src = source.at<cv::Vec4b>(i, j);
                auto& data_img = image.at<cv::Vec3b>(i, j);
                data_img[0] = static_cast<float>(data_src[0]);
                data_img[1] = static_cast<float>(data_src[1]);
                data_img[2] = static_cast<float>(data_src[2]);
                alpha.at<float>(i, j) = static_cast<float>(data_src[3]);
            }
        }
    }
    else {
        throw std::invalid_argument("Source must be a 8UC4 or 32FC4 matrix");
    }    
}

void fuseImage(const cv::Mat& foreground, const cv::Mat& background, const cv::Mat& mask_f, cv::Mat& fusion)
{
    // 1. 检查输入是否为空
    if (foreground.empty() || background.empty() || mask_f.empty()) {
        throw std::invalid_argument("Input matrices cannot be empty");
    }

    // 2. 检查尺寸一致
    if (mask_f.size() != background.size() || mask_f.size() != foreground.size()) {
        throw std::invalid_argument("All input matrices must have the same size");
    }

    // 3. 检查通道数
    if (foreground.channels() != 3 || background.channels() != 3 || mask_f.channels() != 1) {
        throw std::invalid_argument("Foreground and background must be 3-channel; mask must be 1-channel");
    }

    // 4. 检查数据类型
    bool is_uint = bool(foreground.type() == CV_8UC3 && background.type() == CV_8UC3 && mask_f.type() == CV_8UC1);
    bool is_float = bool(foreground.type() == CV_32FC3 && background.type() == CV_32FC3 && mask_f.type() == CV_32FC1);
    if (is_uint == false && is_float == false) {
        throw std::invalid_argument("All inputs must be uint or float");
    }

    // 5. 检查连续性
    if (!foreground.isContinuous() || !background.isContinuous() || !mask_f.isContinuous()) {
        throw std::runtime_error("Inputs must be continuous matrices");
    }

    // 6. 手动循环处理每个像素
    int rows = foreground.rows;
    int cols = foreground.cols;
    size_t total_pixels = rows * cols;
    if (is_uint == true) {
        fusion.create(foreground.size(), CV_8UC3);
        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < foreground.rows; ++i) {
            for (int j = 0; j < foreground.cols; ++j) {
                float alpha = static_cast<float>(mask_f.at<uchar>(i, j)) / 255.f;
                float inv_alpha = 1.0f - alpha;
                for (int c = 0; c < 3; ++c) {
                    float fg_val = static_cast<float>(foreground.at<cv::Vec3b>(i, j)[c]);
                    float bg_val = static_cast<float>(background.at<cv::Vec3b>(i, j)[c]);
                    fusion.at<cv::Vec3b>(i, j)[c] = static_cast<uchar>(fg_val * alpha + bg_val * inv_alpha);
                }
            }
        }
    } 
    else {
        fusion.create(foreground.size(), CV_32FC3);
        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < foreground.rows; ++i) {
            for (int j = 0; j < foreground.cols; ++j) {
                float alpha = static_cast<float>(mask_f.at<float>(i, j)) / 255.f;
                float inv_alpha = 1.0f - alpha;
                for (int c = 0; c < 3; ++c) {
                    float fg_val = static_cast<float>(foreground.at<cv::Vec3b>(i, j)[c]);
                    float bg_val = static_cast<float>(background.at<cv::Vec3b>(i, j)[c]);
                    fusion.at<cv::Vec3b>(i, j)[c] = static_cast<float>(fg_val * alpha + bg_val * inv_alpha);
                }
            }
        }
    }
}