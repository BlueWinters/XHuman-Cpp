
#ifndef __Face_Render__
#define __Face_Render__

#include "singleton.h"
#include "tools/ximage.h"
#include "face_3dmm.h"


struct Face3DMMCoefficientsMatrix
{
    cv::Mat identity;
    cv::Mat expression;
    cv::Mat texture;
    cv::Mat angles;
    cv::Mat gamma;
    cv::Mat translation;
};

struct FaceParameter
{
    cv::Mat face_shape;       // 1,107127 --> 35709,3
    cv::Mat rotation;         // 3,3
    cv::Mat face_vertex;      // 35709,3
    cv::Mat face_texture;     // 1,107127 --> 35709,3
    cv::Mat face_norm_roted;  // 35709,3
    cv::Mat gray_shading;     // 35709,3
};

struct FaceRenderResult 
{
    cv::Mat image;  // 渲染图像
    cv::Mat mask;   // 掩码
    cv::Mat depth;  // 深度图
};


class FaceRender
{
public:
    THREAD_SAFE_SINGLETON_AUTOMATIC(FaceRender);
public:
    FaceRender();
    ~FaceRender();

public:
    cv::Mat persc_proj;
    cv::Mat light_direction;
    float light_direction_norm[5];
    cv::Mat light_intensities;    
    // bfm
    cv::Mat mean_shape;
    cv::Mat id_base;
    cv::Mat exp_base;
    cv::Mat tex_mean;
    cv::Mat tex_base;
    cv::Mat point_buf;
    cv::Mat tri;
    cv::Mat key_points;
    // uv
    cv::Mat bfm_uv;  // 35709, 2
public:
    const float fov = 12.593637f;
    const int rast_h = 224;
    const int rast_w = 224;
    const float camera_distance = 10.f;
    const float ndc_proj[16] = { 9.06250f, 0.f, 0.f, 0.f, 0.f, 9.06250f, 0.f, 0.f, 0.f, 0.f, 2.f, 1.f, 0.f, 0.f, -15.f, 0.f };

public:
    void initialize(const char* path_bfm);
    void inference(const Face3DMMResult& result_3dmm, const cv::Mat& uv_texture, FaceRenderResult& result_render);
    void inference(const Face3DMMResult& result_3dmm, FaceRenderResult& result_render);
    void pasteBack(const Face3DMMResult& result_3dmm, const FaceRenderResult& result_render, const cv::Mat& source, FaceRenderResult& result_source);
protected:
    void calculateParameters(const Face3DMMCoefficients& coefficients, FaceParameter& param, bool with_norm = false);
    void transformToMatrix(const Face3DMMCoefficients& coefficients, Face3DMMCoefficientsMatrix& matrix);
    void computeShape(Face3DMMCoefficientsMatrix& coefficients, FaceParameter& param);
    void computeRotation(Face3DMMCoefficientsMatrix& coefficients, FaceParameter& param);
    void transform(Face3DMMCoefficientsMatrix& coefficients, FaceParameter& param);
    void toCamera(Face3DMMCoefficientsMatrix& coefficients, FaceParameter& param);
    void computeTexture(Face3DMMCoefficientsMatrix& coefficients, FaceParameter& param);
    void computeNorm(Face3DMMCoefficientsMatrix& coefficients, FaceParameter& param);
protected:
    void normalizeDepth(const cv::Mat& depth, FaceRenderResult& result);
protected:
    void renderWithTexture(FaceParameter& param, const cv::Mat& uv_texture, FaceRenderResult& result);
protected:
    void renderShape(FaceParameter& param, FaceRenderResult& result);
    void computeGrayShadingWithDirectionLight(FaceParameter& param);
};


#endif