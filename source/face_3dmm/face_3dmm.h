
#ifndef __Face3DMM__
#define __Face3DMM__

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <ncnn/net.h>
#include <ncnn/mat.h>
#include "singleton.h"
#include "tools/ximage.h"
#include "face_base/face_tracking.h"


struct FormatInfo
{
    // w0 * s, h0 * s
    int h, w;
    // padding
    int pad_w, pad_h;
    // face box
    int lft, top, rig, bot;
    int landmark[136];
    // clip image
    cv::Mat clip;
};

union Face3DMMCoefficients
{
    // 80+64+80+3+27+3 = 257
    float parameters[257];
    struct
    {
        float identity[80];
        float expression[64];
        float texture[80];
        float angles[3];
        float gamma[27];
        float translation[3];
    };
};

struct Face3DMMResult
{
    // format information
    FormatInfo format_info;
    // 3dmm coefficient
    Face3DMMCoefficients coefficients;
};

typedef std::vector<Face3DMMResult> Face3DMMResultVector;


class Face3DMM
{
public:
    THREAD_SAFE_SINGLETON_AUTOMATIC(Face3DMM);
public:
    Face3DMM();
    ~Face3DMM();

protected:
    int num_threads = 2;
    bool light_mode = false;
    bool use_gpu = true;
    ncnn::Net net;
protected:
    const int target_size = 224;
    float rescale_factor = 102.f;
    FaceTracking face_tracker;

public:
    void initialize();
    void initialize(const char* path_param, const char* path_bin);
    void inference(XImage& image, Face3DMMResultVector& result_vector);
protected:
    void formatInput(const cv::Mat& image, const int* landmarks, cv::Mat& image_cropped, FormatInfo& format_info);
    void calculate5Points(const int* landmark, cv::Mat& mat, int height);
    void calculateParameters(const cv::Mat& xp, const float* x, float* t, float& s);
    void cropImage(const cv::Mat& image_bgr, cv::Mat& image_cropped, 
        const float* t, const float s, const int target_size, FormatInfo& format_info);
    void forward(cv::Mat &input, Face3DMMResult& result);
};

#endif