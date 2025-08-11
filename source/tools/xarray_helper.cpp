
#include "xarray_helper.h"


void transformXArray2Matrix(const XArray& array, cv::Mat& mat)
{
    size_t rows = 0, cols = 0;
    if (array.shape.size() == 1) {
        rows = array.shape[0];
        cols = 1;
    }
    else if (array.shape.size() == 2) {
        rows = array.shape[0];
        cols = array.shape[1];
    }
    else {
        throw std::invalid_argument("Shape must be 1 or 2 dimensions.");
    }

    if (array.type_info.type == DataType::FLOAT32) {
        mat.create(rows, cols, CV_32F);
        std::memcpy(mat.data, array.data, array.num_bytes);
    }
    else if (array.type_info.type == DataType::INT32) {
        mat.create(rows, cols, CV_32S);
        std::memcpy(mat.data, array.data, array.num_bytes);
    }
    else {
        throw std::invalid_argument("Unsupported data type.");
    }
}

bool int32fromFile(const char* path, cv::Mat& mat)
{
    XArray array;
    if (array.load(path) == false) {
        return false;
    }

    const auto& shape = array.shape;
    const void* buffer = array.data;

    if (shape.empty()) {
        throw std::invalid_argument("Shape cannot be empty.");
    }
    if (shape.size() > 2) {
        throw std::invalid_argument("Dimentionin must be 1 or 2.");
    }

    unsigned int num_elements = array.getNumElements();
    if (num_elements <= 0) {
        throw std::invalid_argument("Invalid shape: total elements <= 0");
    }

    int rows, cols;
    if (shape.size() == 1) {
        // 1D: use as single row (1, N) - common for coefficients
        rows = 1;
        cols = shape[0];
    }
    else {
        // 2D or higher: treat first dim as rows, rest as cols (flattened)
        rows = shape[0];
        cols = shape[1];
    }

    // Create cv::Mat and copy data
    mat.create(rows, cols, CV_32S);  // CV_32F for int32
    std::memcpy(mat.data, buffer, num_elements * sizeof(int));
    return true;
}

bool float32fromFile(const char* path, cv::Mat& mat)
{
    XArray array;
    if (array.load(path) == false) {
        return false;
    }

    const auto& shape = array.shape;
    const void* buffer = array.data;

    if (shape.empty()) {
        throw std::invalid_argument("Shape cannot be empty.");
    }
    if (shape.size() > 2) {
        throw std::invalid_argument("Dimentionin must be 1 or 2.");
    }

    unsigned int num_elements = array.getNumElements();
    if (num_elements <= 0) {
        throw std::invalid_argument("Invalid shape: total elements <= 0");
    }

    int rows, cols;
    if (shape.size() == 1) {
        // 1D: use as single row (1, N) - common for coefficients
        rows = 1;
        cols = shape[0];
    }
    else {
        // 2D or higher: treat first dim as rows, rest as cols (flattened)
        rows = shape[0];
        cols = shape[1];
    }

    // Create cv::Mat and copy data
    mat.create(rows, cols, CV_32F);  // CV_32F for float32
    std::memcpy(mat.data, buffer, num_elements * sizeof(float));
    return true;
}
