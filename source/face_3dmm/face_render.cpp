
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "face_render.h"
#include "mesh_render.h"
#include "tools/xarray_helper.h"
#include "tools/cvfunc.h"


FaceRender::FaceRender()
{
    
}

FaceRender::~FaceRender()
{

}

void FaceRender::initialize(const char* path_bfm)
{
    persc_proj = (cv::Mat_<float>(3, 3) << 
        1015.f, 0.f, 0.f, 
        0.f, 1015.f, 0.f, 
        112.f, 112.f, 1.f);
    light_direction = (cv::Mat_<float>(5, 3) <<
       -1.f, +1.f, +1.f,  
       +1.f, +1.f, +1.f,  
       -1.f, -1.f, +1.f,  
       +1.f, -1.f, +1.f,  
       +0.f, +0.f, +1.f);
   light_intensities = (cv::Mat_<float>(5, 3) <<
       +1.7f, +1.7f, +1.7f,
       +1.7f, +1.7f, +1.7f,
       +1.7f, +1.7f, +1.7f,
       +1.7f, +1.7f, +1.7f,
       +1.7f, +1.7f, +1.7f);
   for (int i = 0; i < 5; i++) {
       light_direction_norm[i] = cv::norm(this->light_direction.row(i));
   }

   XArrayContainer container;
   container.load(path_bfm);
   transformXArray2Matrix(container["bfm_uv"], bfm_uv);
   transformXArray2Matrix(container["mean_shape"], mean_shape);
   transformXArray2Matrix(container["id_base"], id_base);
   transformXArray2Matrix(container["exp_base"], exp_base);
   transformXArray2Matrix(container["tex_mean"], tex_mean);
   transformXArray2Matrix(container["tex_base"], tex_base);
   transformXArray2Matrix(container["point_buf"], point_buf);
   transformXArray2Matrix(container["tri"], tri);
   transformXArray2Matrix(container["key_points"], key_points);

    id_base = id_base.t();                  // 80,107127
    exp_base = exp_base.t();                // 64,107127
    mean_shape = mean_shape.reshape(0, 1);  // 1,107127
    tex_mean = tex_mean.reshape(0, 1);      // 1,107127
    tex_base = tex_base.t();                // 80,107127
    // inplace operation: mat.col return a data view
    cv::Mat last_col = bfm_uv.col(1);
    last_col = 1.0f - last_col;
}

void FaceRender::computeShape(Face3DMMCoefficientsMatrix& coefficients, FaceParameter& param)
{
    cv::Mat id = coefficients.identity * this->id_base;
    cv::Mat exp = coefficients.expression * this->exp_base;
    param.face_shape = id + exp + this->mean_shape;
    std::vector<int> new_shape = { 35709, 3 };
    param.face_shape = param.face_shape.reshape(0, new_shape);
}

void FaceRender::computeRotation(Face3DMMCoefficientsMatrix& coefficients, FaceParameter& param)
{
    const cv::Mat& angles = coefficients.angles;
    float x = angles.ptr<float>(0)[0];
    float y = angles.ptr<float>(0)[1];
    float z = angles.ptr<float>(0)[2];
    //const float cos_x = std::cos(x), sin_x = std::sin(x);
    //const float cos_y = std::cos(y), sin_y = std::sin(y);
    //const float cos_z = std::cos(z), sin_x = std::sin(z);
    cv::Mat rot_x = (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, cos(x), -sin(x), 0, sin(x), cos(x));
    cv::Mat rot_y = (cv::Mat_<float>(3, 3) << cos(y), 0, sin(y), 0, 1, 0, -sin(y), 0, cos(y));
    cv::Mat rot_z = (cv::Mat_<float>(3, 3) << cos(z), -sin(z), 0, sin(z), cos(z), 0, 0, 0, 1);

    cv::Mat rot = rot_z * rot_y * rot_x;
    param.rotation = rot.t();
}

void FaceRender::transform(Face3DMMCoefficientsMatrix& coefficients, FaceParameter& param)
{
    // c++里没有numpy的广播机制,所以这里需要用repeat
    cv::Mat translation = cv::repeat(coefficients.translation, param.face_shape.size[0], 1);
    param.face_vertex = param.face_shape * param.rotation + translation;
}

void FaceRender::toCamera(Face3DMMCoefficientsMatrix& coefficients, FaceParameter& param)
{
    cv::Mat& face_vertex = param.face_vertex;
    face_vertex.col(2) = this->camera_distance - face_vertex.col(2);
}

void FaceRender::computeTexture(Face3DMMCoefficientsMatrix& coefficients, FaceParameter& param)
{
    cv::Mat tex = coefficients.texture * this->tex_base;
    cv::Mat face_texture = tex + this->tex_mean;
    face_texture.convertTo(param.face_texture, CV_32F, 1.0f / 255.f);
    std::vector<int> new_shape = { 35709, 3 };
    param.face_texture = param.face_texture.reshape(0, new_shape);
}

void FaceRender::computeNorm(Face3DMMCoefficientsMatrix& coefficients, FaceParameter& param)
{
    cv::Mat& face_shape = param.face_shape;
    const int tri_shape[] = { 70789, 3 };
    cv::Mat face_norm = cv::Mat::zeros(tri_shape[0] + 1, 3, CV_32FC1);
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < tri_shape[0]; i++) {
        const cv::Mat v1 = face_shape.row(tri.ptr<int>(i, 0)[0]);
        const cv::Mat v2 = face_shape.row(tri.ptr<int>(i, 1)[0]);
        const cv::Mat v3 = face_shape.row(tri.ptr<int>(i, 2)[0]);
        cv::Mat e1 = v1 - v2;
        cv::Mat e2 = v2 - v3;
        cv::Mat n = e1.cross(e2);
        face_norm.row(i) = n / cv::norm(n);
    }

    const int point_buf_shape[] = { 35709, 8 };
    cv::Mat vertex_norm = cv::Mat::zeros(point_buf_shape[0], 3, CV_32FC1);
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < point_buf_shape[0]; i++) {
        cv::Mat vs = cv::Mat::zeros(point_buf_shape[1], 3, CV_32FC1);
        for (int j = 0; j < point_buf_shape[1]; j++) {
            face_norm.row(point_buf.ptr<int>(i, j)[0]).copyTo(vs.row(j));
        }
        cv::Mat vs_colsum;
        cv::reduce(vs, vs_colsum, 0, cv::REDUCE_SUM, CV_32FC1);   // 沿着列求和
        vertex_norm.row(i) = vs_colsum / cv::norm(vs_colsum);
    }

    param.face_norm_roted = vertex_norm * param.rotation;
}

void FaceRender::computeGrayShadingWithDirectionLight(FaceParameter& param)
{
    const cv::Mat& face_texture = cv::Mat::ones(35709, 3, CV_32FC1) * 0.78; // param.face_texture(35709,3)
    const cv::Mat& normals = param.face_norm_roted;
    param.gray_shading = cv::Mat(normals.rows, 3, CV_32FC1);
    cv::Mat& gray_shading = param.gray_shading;
    float gray_shading_max = 0;
    for (int i = 0; i < normals.rows; i++) {
        float sum_[3] = { 0.f, 0.f, 0.f };
        for (int j = 0; j < 5; j++) {
            const float x = light_direction_norm[j];
            float y = cv::sum(normals.row(i).mul(this->light_direction.row(j) / x))[0];
            y = std::max(std::min(y, 1.f), 0.f);
            sum_[0] += (y * this->light_intensities.ptr<float>(j)[0]);
            sum_[1] += (y * this->light_intensities.ptr<float>(j)[1]);
            sum_[2] += (y * this->light_intensities.ptr<float>(j)[2]);
        }
        gray_shading.ptr<float>(i)[0] = face_texture.ptr<float>(i)[0] * (sum_[0] / 5.f);
        gray_shading.ptr<float>(i)[1] = face_texture.ptr<float>(i)[1] * (sum_[1] / 5.f);
        gray_shading.ptr<float>(i)[2] = face_texture.ptr<float>(i)[2] * (sum_[2] / 5.f);
        auto value = gray_shading.ptr<float>(i)[0];
        gray_shading_max = StdMax(gray_shading_max, value);
    }
}

void FaceRender::transformToMatrix(const Face3DMMCoefficients& coefficients, Face3DMMCoefficientsMatrix& matrix)
{
    formatAsMatrix(coefficients.identity, 1, 80, matrix.identity);
    formatAsMatrix(coefficients.expression, 1, 64, matrix.expression);
    formatAsMatrix(coefficients.texture, 1, 80, matrix.texture);
    formatAsMatrix(coefficients.angles, 3, 1, matrix.angles);
    formatAsMatrix(coefficients.gamma, 27, 1, matrix.gamma);
    formatAsMatrix(coefficients.translation, 1, 3, matrix.translation);
}

void FaceRender::calculateParameters(const Face3DMMCoefficients& coefficients, FaceParameter& param, bool with_norm)
{
    Face3DMMCoefficientsMatrix coefficients_mat;
    transformToMatrix(coefficients, coefficients_mat);
    computeShape(coefficients_mat, param);
    computeRotation(coefficients_mat, param);
    transform(coefficients_mat, param);
    toCamera(coefficients_mat, param);
    // only for shape
    if (with_norm) {
        computeNorm(coefficients_mat, param);
    }
}
void FaceRender::renderWithTexture(FaceParameter& param, const cv::Mat& uv_texture, FaceRenderResult& result)
{
    cv::Mat& vertex = param.face_vertex;
    vertex.col(1) = 0.f - vertex.col(1);

    vertex = vertex.isContinuous() ? vertex : vertex.clone();
    tri = tri.isContinuous() ? tri : tri.clone();
    bfm_uv = bfm_uv.isContinuous() ? bfm_uv : bfm_uv.clone();
    
    // 光栅化
    cv::Mat rast_out = cv::Mat::zeros(rast_h, rast_w, CV_32FC4); // [h, w, 4]
    render_rasterize(vertex.ptr<float>(), vertex.rows, tri.ptr<int>(), tri.rows,
        ndc_proj, rast_h, rast_w, rast_out.ptr<float>());

    // 插值UV坐标
    cv::Mat interp_out = cv::Mat::zeros(rast_h, rast_w, CV_32FC2); // [h, w, 2]
    render_interpolate(bfm_uv.ptr<float>(), bfm_uv.rows, bfm_uv.cols,
        rast_out.ptr<float>(), rast_h, rast_w, tri.ptr<int>(), tri.rows, interp_out.ptr<float>());

    // 纹理采样
    cv::Mat image_float = cv::Mat::zeros(rast_h, rast_w, CV_32FC4);
    cv::Mat uv_texture_float;
    uv_texture.convertTo(uv_texture_float, CV_32F);  // 确保uv_texture是float32类型
    render_texture(uv_texture_float.ptr<float>(), uv_texture_float.rows, uv_texture_float.cols, 4,
        interp_out.ptr<float>(), rast_h, rast_w, image_float.ptr<float>());
    image_float.convertTo(result.image, CV_8U);

    // 生成mask
    cv::Mat alpha_channel;
    cv::extractChannel(rast_out, alpha_channel, 3);  // BGRA
    // np.where(rast_out[..., 3] > 0, 1, 0).astype(np.uint8)
    cv::Mat where_mat;
    cv::compare(alpha_channel, cv::Scalar(0.0f), where_mat, cv::CMP_GT);
    where_mat.convertTo(result.mask, CV_8UC1, 255.f, 0.f);

    // 插值深度
    cv::Mat vertex_z = vertex.col(2).clone();
    cv::Mat depth_out = cv::Mat::zeros(rast_h, rast_w, CV_32FC1); // [h, w, 1]
    render_interpolate(vertex_z.ptr<float>(), vertex_z.rows, vertex_z.cols,
        rast_out.ptr<float>(), rast_h, rast_w, tri.ptr<int>(), tri.rows, depth_out.ptr<float>());
    normalizeDepth(depth_out, result);
}

void FaceRender::renderShape(FaceParameter& param, FaceRenderResult& result)
{
    computeGrayShadingWithDirectionLight(param);

    cv::Mat& vertex = param.face_vertex;
    vertex.col(1) = 0.f - vertex.col(1);

    vertex = vertex.isContinuous() ? vertex : vertex.clone();
    tri = tri.isContinuous() ? tri : tri.clone();

    // 光栅化
    cv::Mat rast_out = cv::Mat::zeros(rast_h, rast_w, CV_32FC4); // [h, w, 4]
    render_rasterize(vertex.ptr<float>(), vertex.rows, tri.ptr<int>(), tri.rows,
        ndc_proj, rast_h, rast_w, rast_out.ptr<float>());

    // 插值
    cv::Mat shape = cv::Mat::zeros(rast_h, rast_w, CV_32FC3); // [h, w, 2]
    render_interpolate(param.gray_shading.ptr<float>(), param.gray_shading.rows, param.gray_shading.cols,
        rast_out.ptr<float>(), rast_h, rast_w, tri.ptr<int>(), tri.rows, shape.ptr<float>());
    shape.convertTo(result.image, CV_8UC3, 255.f, 0.f);

    // 生成mask
    cv::Mat alpha_channel;
    cv::extractChannel(rast_out, alpha_channel, 3);  // BGRA
    // np.where(rast_out[..., 3] > 0, 1, 0).astype(np.uint8)
    cv::Mat where_mat;
    cv::compare(alpha_channel, cv::Scalar(0.0f), where_mat, cv::CMP_GT);
    where_mat.convertTo(result.mask, CV_8UC1, 255.f, 0.f);

    // 插值深度
    cv::Mat vertex_z = vertex.col(2).clone();
    cv::Mat depth_out = cv::Mat::zeros(rast_h, rast_w, CV_32FC1); // [h, w, 1]
    render_interpolate(vertex_z.ptr<float>(), vertex_z.rows, vertex_z.cols,
        rast_out.ptr<float>(), rast_h, rast_w, tri.ptr<int>(), tri.rows, depth_out.ptr<float>());
    normalizeDepth(depth_out, result);
}

void FaceRender::normalizeDepth(const cv::Mat& depth, FaceRenderResult& result)
{
    const cv::Mat& mask = result.mask;

    int rows = depth.rows;
    int cols = depth.cols;

    // 第一次遍历：找非零区域的 min 和 max
    float d_min = FLT_MAX;
    float d_max = -FLT_MAX;
    bool found = false;

    const float* d_data = depth.ptr<float>(0);
    const uchar* m_data = mask.ptr<uchar>(0);

    for (int i = 0; i < rows * cols; ++i) {
        if (m_data[i] > 0 && d_data[i] > 0) {
            if (d_data[i] < d_min) d_min = d_data[i];
            if (d_data[i] > d_max) d_max = d_data[i];
            found = true;
        }
    }

    // 如果没有有效点
    if (!found || d_max <= d_min) {
        result.depth = cv::Mat::zeros(rows, cols, CV_8UC1);
        return;
    }

    float range = d_max - d_min;

    // 第二次遍历：归一化 + 裁剪 + mask 处理
    result.depth = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat& result_depth = result.depth;
    for (int i = 0; i < result_depth.rows; ++i) {
        for (int j = 0; j < result_depth.cols; ++j) {
            if (mask.ptr<uchar>(i, j)[0] > 0) {
                float val = (depth.ptr<float>(i, j)[0] - d_min) / range * 255.0f;
                val = StdMax(val, 0.f);
                val = StdMin(val, 255.f);
                val = 255.0f - val;
                result_depth.at<uchar>(i, j) = static_cast<uchar>(val);
            }
            else {
                result_depth.at<uchar>(i, j) = 0;
            }
        }
    }
}

void FaceRender::inference(const Face3DMMResult& result_3dmm, const cv::Mat& uv_texture, FaceRenderResult& result_render)
{
    FaceParameter face_param;
    calculateParameters(result_3dmm.coefficients, face_param, false);
    renderWithTexture(face_param, uv_texture, result_render);
}

void FaceRender::inference(const Face3DMMResult& result_3dmm, FaceRenderResult& result_render)
{
    FaceParameter face_param;
    calculateParameters(result_3dmm.coefficients, face_param, true);
    renderShape(face_param, result_render);
}

void FaceRender::pasteBack(const Face3DMMResult& result_3dmm, const FaceRenderResult& result_render, const cv::Mat& source, FaceRenderResult& result_source)
{
    // 解包 box 值
    auto& format_info = result_3dmm.format_info;
    int hh = format_info.h, ww = format_info.w;
    int lft = format_info.lft, top = format_info.top, rig = format_info.rig, bot = format_info.bot;

    // 获取 bgr 的尺寸
    int h = source.rows;
    int w = source.cols;
    int c = source.channels();

    // 计算缩放比例
    float rh = static_cast<float>(hh) / h;
    float rw = static_cast<float>(ww) / w;

    // 计算新的尺寸
    const cv::Mat& face = result_render.image;
    int nh = static_cast<int>(std::round(face.rows / rh));
    int nw = static_cast<int>(std::round(face.cols / rw));

    // 计算填充参数
    int lp = std::max(static_cast<int>(std::round(lft / rw)), 0);
    int rp_src = w - nw - lp;
    int rp = std::max(rp_src, 0);
    int tp = std::max(static_cast<int>(std::round(top / rh)), 0);
    int bp_src = h - nh - tp;
    int bp = std::max(bp_src, 0);

    cv::Mat face_new, face_padded;
    cv::resize(face, face_new, cv::Size(nw, nh), 0, 0, cv::INTER_CUBIC);
    cv::copyMakeBorder(face_new, face_padded, tp, bp, lp, rp, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    if (result_render.mask.empty() == false) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat mask_eroded, mask_resized, mask_padded;
        cv::erode(result_render.mask, mask_eroded, kernel);
        cv::resize(mask_eroded, mask_resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
        cv::copyMakeBorder(mask_resized, mask_padded, tp, bp, lp, rp, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        if (rp_src < 0) {
            mask_padded = mask_padded(cv::Range::all(), cv::Range(0, mask_padded.cols + rp_src)).clone();
        }
        if (bp_src < 0) {
            mask_padded = mask_padded(cv::Range(0, mask_padded.rows + bp_src), cv::Range::all()).clone();
        }
        result_source.mask = mask_padded.clone();
    }
    
    if (result_render.depth.empty() == false) {
        cv::Mat depth_resized, depth_padded;
        cv::resize(result_render.depth, depth_resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
        cv::copyMakeBorder(depth_resized, depth_padded, tp, bp, lp, rp, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        if (rp_src < 0) {
            depth_padded = depth_padded(cv::Range::all(), cv::Range(0, depth_padded.cols + rp_src)).clone();
        }
        if (bp_src < 0) {
            depth_padded = depth_padded(cv::Range(0, depth_padded.rows + bp_src), cv::Range::all()).clone();
        }
        result_source.depth = depth_padded.clone();
    }

    // 处理负的填充情况（裁剪）
    if (rp_src < 0) {
        face_padded = face_padded(cv::Range::all(), cv::Range(0, face_padded.cols + rp_src));
    }
    if (bp_src < 0) {
        face_padded = face_padded(cv::Range(0, face_padded.rows + bp_src), cv::Range::all());
    }

    // 创建结果图像
    cv::Mat composite = cv::Mat::zeros(source.size(), CV_32FC3);

    if (face_padded.channels() == 4) {
        // 通道分离
        cv::Mat face_bgr, face_alpha;
        splitImage(face_padded, face_bgr, face_alpha);
        // 将非255的alpha值设为0
        cv::Mat mask_alpha;
        cv::compare(face_alpha, 255, mask_alpha, cv::CMP_EQ);
        face_alpha.setTo(0, ~mask_alpha);
        // 高斯模糊
        const int k = 7;
        cv::Mat alpha_blurred;
        cv::GaussianBlur(mask_alpha, alpha_blurred, cv::Size(k, k), k / 2.0, k / 2.0);
        fuseImage(face_bgr, source, alpha_blurred, result_source.image);
    }
    else if (face_padded.channels() == 3) {
        fuseImage(face_padded, source, result_source.mask, result_source.image);
    }
    else {
        throw std::runtime_error("Unsupported number of channels in face image.");
    }
}