
#include <iostream>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <cassert>
#include "xarray.h"


// 默认构造函数
XArray::XArray() : type_info(), num_bytes(0), data(nullptr) 
{
    // 初始化为默认值
}

// 带参数的构造函数
XArray::XArray(const std::vector<unsigned int>& shape, DataType data_type, void* data, bool copy_data)
    : shape(shape), num_bytes(0), data(nullptr) 
{
    initialize(shape, data_type, data, copy_data);
}

// 析构函数
XArray::~XArray()
{
    clear();
}

// 清理数据
void XArray::clear() 
{
    if (data) {
        free(data);
        data = nullptr;
    }
    shape.clear();
    num_bytes = 0;
    type_info = DataTypeInfo(DataType::NONE, 0, "none");
}

void XArray::initialize(const std::vector<unsigned int>& shape, DataType data_type, void* data, bool copy_data)
{
    clear();

    // 验证形状
    this->shape = shape;
    size_t total_elements = 1;
    for (auto dim : this->shape) {
        total_elements *= static_cast<size_t>(dim);
        if (dim <= 0) {
            throw std::invalid_argument("Invalid dimension size");
        }
    }

    // 从 DataTypeMap 获取类型信息
    auto it = DataTypeMap.find(static_cast<uint32_t>(data_type));
    if (it != DataTypeMap.end()) {
        this->type_info = it->second;
    }
    else {
        // 如果类型不在映射表中
        throw std::invalid_argument("Invalid data type");
    }

    // 计算总字节数
    this->num_bytes = static_cast<unsigned int>(total_elements * this->type_info.size);

    // 分配内存
    if (this->num_bytes > 0) {
        if (data != nullptr) {
            if (copy_data) {
                this->data = malloc(this->num_bytes);
                if (!this->data) {
                    throw std::bad_alloc();
                }
                // 拷贝数据
                std::memcpy(this->data, data, num_bytes);
            }
            else {
                // 赋值指针
                this->data = data;
            }
        }
        else {
            // 创造数组并初始化为0
            this->data = malloc(this->num_bytes);
            std::memset(this->data, 0, this->num_bytes);
        }
    }
    else {
        throw std::invalid_argument("Unknown exception");
    }
}

// 获取维度数
int XArray::dimensions() const 
{
    return static_cast<int>(shape.size());
}

// 获取总的元素数量
size_t XArray::size() const 
{
    if (shape.empty()) {
        return 0;
    }

    size_t total_elements = 1;
    for (auto dim : shape) {
        total_elements *= static_cast<size_t>(dim);
    }
    return total_elements;
}

// 输出基本信息
void XArray::printInfo() const 
{
    if (shape.empty()) {
        std::cout << "[]" << std::endl;
        return;
    }

    std::cout << "dtype=" << type_info.name << ";";
    std::cout << "shape=(";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) {
            std::cout << ",";
        }
    }
    std::cout << ")" << std::endl;
}

// 输出数组信息
void XArray::print() const {
    if (!data || shape.empty()) {
        std::cout << "[]" << std::endl;
        return;
    }

    // 根据维度数选择打印方式
    switch (shape.size()) {
    case 1:
        print1D();
        break;
    case 2:
        print2D();
        break;
    case 3:
        print3D();
        break;
    }
}

// 获取元素大小
size_t XArray::getElementSize() const 
{
    return type_info.size;
}

// 获取元素个数
size_t XArray::getNumElements() const
{
    size_t total_elements = 1;
    for (auto dim : shape) {
        total_elements *= static_cast<size_t>(dim);
    }
    return total_elements;
}

// 打印单个元素
void XArray::printElement(size_t index) const 
{
    if (index >= size() || !data) return;

    size_t element_size = getElementSize();
    char* ptr = static_cast<char*>(data) + index * element_size;

    switch (type_info.type) {
    case DataType::FLOAT32:
        std::cout << *reinterpret_cast<float*>(ptr);
        break;
    case DataType::FLOAT64:
        std::cout << *reinterpret_cast<double*>(ptr);
        break;
    case DataType::INT32:
        std::cout << *reinterpret_cast<int*>(ptr);
        break;
    case DataType::INT64:
        std::cout << *reinterpret_cast<int64_t*>(ptr);
        break;
    case DataType::UINT8:
        std::cout << static_cast<int>(*reinterpret_cast<uint8_t*>(ptr));
        break;
    case DataType::INT16:
        std::cout << *reinterpret_cast<int16_t*>(ptr);
        break;
    case DataType::FLOAT16:
        throw std::runtime_error("Not implement error");
        break;
    default:
        std::cout << "?";
        break;
    }
}

// 输出数组信息(一维)
void XArray::print1D() const 
{
    std::cout << "[";
    for (size_t i = 0; i < size(); ++i) {
        if (i > 0) {
            std::cout << " ";
        }

        printElement(i);

        if (i < size() - 1) {
            std::cout << ',';
        }
    }
    std::cout << "]" << std::endl;
}

// 输出数组信息(二维)
void XArray::print2D() const 
{
    if (shape.size() < 2) return;

    unsigned int rows = shape[0];
    unsigned int cols = shape[1];

    for (unsigned int i = 0; i < rows; ++i) {
        if (i == 0) {
            std::cout << "[";
        }
        else {
            std::cout << " ";
        }

        std::cout << "[";
        for (unsigned int j = 0; j < cols; ++j) {
            if (j > 0) std::cout << " ";

            auto index = i * cols + j;
            printElement(index);

            if (j < cols - 1) {
                std::cout << ',';
            }
        }
        std::cout << "]";

        if (i == rows - 1) {
            std::cout << "]" << std::endl;
        }
        else {
            std::cout << "," << std::endl;
        }
    }
}

// 输出数组信息(三维)
void XArray::print3D() const 
{
    if (shape.size() < 3) return;

    unsigned int dim0 = shape[0];
    unsigned int dim1 = shape[1];
    unsigned int dim2 = shape[2];

    for (unsigned int i = 0; i < dim0; ++i) {
        if (i == 0) {
            std::cout << "[";
        }
        else {
            std::cout << " ";
        }

        std::cout << "[";
        for (unsigned int j = 0; j < dim1; ++j) {
            if (j > 0) std::cout << ", " << std::endl << "  ";
            std::cout << "[";
            for (unsigned int k = 0; k < dim2; ++k) {
                if (k > 0) {
                    std::cout << ", ";
                }
                auto index = i * dim1 * dim2 + j * dim2 + k;
                printElement(index);
            }
            std::cout << "]";
        }
        std::cout << "]";

        if (i == dim0 - 1) {
            std::cout << "]" << std::endl;
        }
        else {
            std::cout << "," << std::endl;
        }
    }
}

// 从文件读取
bool XArray::load(const std::string& path) 
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    try {
        // 读取版本号
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) {
            throw std::runtime_error("Unsupported file version");
        }

        // 读取数据类型代码
        uint32_t dtype_code;
        file.read(reinterpret_cast<char*>(&dtype_code), sizeof(dtype_code));

        // 检查类型映射是否存在
        auto it = DataTypeMap.find(dtype_code);
        if (it == DataTypeMap.end()) {
            throw std::runtime_error("Unknown data type code: " + std::to_string(dtype_code));
        }

        // 设置类型信息
        type_info = it->second;

        // 读取维度数
        uint32_t dimensions;
        file.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));

        // 读取每个维度的大小
        shape.resize(dimensions);
        size_t total_elements = 1;

        for (uint32_t i = 0; i < dimensions; ++i) {
            uint32_t dim_size;
            file.read(reinterpret_cast<char*>(&dim_size), sizeof(dim_size));
            shape[i] = static_cast<unsigned int>(dim_size);
            total_elements *= dim_size;
        }

        // 释放旧数据
        if (data) {
            free(data);
            data = nullptr;
        }

        // 分配新内存
        num_bytes = static_cast<unsigned int>(total_elements * type_info.size);
        data = malloc(num_bytes);
        if (!data) {
            throw std::bad_alloc();
        }

        // 读取数据
        file.read(static_cast<char*>(data), num_bytes);

        if (file.gcount() != static_cast<std::streamsize>(num_bytes)) {
            clear();
            return false;
        }

        file.close();
        return true;

    }
    catch (const std::exception& e) {
        clear();
        file.close();
        throw;
    }
}


void XArrayContainer::addArray(const std::string& key, const XArray& array) 
{
    array_map[key] = array;
}

bool XArrayContainer::getArray(const std::string& key, XArray& out_array) const 
{
    auto it = array_map.find(key);
    if (it != array_map.end()) {
        out_array = it->second;
        return true;
    }
    return false;
}

void XArrayContainer::removeArray(const std::string& key) 
{
    array_map.erase(key);
}

bool XArrayContainer::hasArray(const std::string& key) const 
{
    return array_map.find(key) != array_map.end();
}

std::vector<std::string> XArrayContainer::keys() const 
{
    std::vector<std::string> keys;
    for (const auto& pair : array_map) {
        keys.push_back(pair.first);
    }
    return keys;
}

void XArrayContainer::printAll() const 
{
    for (const auto& pair : array_map) {
        std::cout << "name=" << pair.first << ";";
        const XArray& array = pair.second;
        std::cout << "dtype=" << array.type_info.name << ";";
        std::cout << "shape=(";
        for (size_t i = 0; i < array.shape.size(); ++i) {
            std::cout << array.shape[i];
            if (i < array.shape.size() - 1) {
                std::cout << ",";
            }
        }
        std::cout << ")" << std::endl;
    }
}

// 自定义二进制格式
bool XArrayContainer::save(const std::string& path) const 
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    size_t num_arrays = array_map.size();
    file.write(reinterpret_cast<const char*>(&num_arrays), sizeof(size_t));

    for (const auto& pair : array_map) {
        // 写 key 长度和字符串
        size_t key_len = pair.first.length();
        file.write(reinterpret_cast<const char*>(&key_len), sizeof(size_t));
        file.write(pair.first.c_str(), key_len);

        // 写 shape
        size_t dim = pair.second.shape.size();
        file.write(reinterpret_cast<const char*>(&dim), sizeof(size_t));
        for (size_t i = 0; i < dim; ++i) {
            file.write(reinterpret_cast<const char*>(&pair.second.shape[i]), sizeof(unsigned int));
        }

        // 写 type
        int type_code = static_cast<int>(pair.second.type_info.type);
        file.write(reinterpret_cast<const char*>(&type_code), sizeof(int));

        // 写数据
        file.write(reinterpret_cast<const char*>(pair.second.data), pair.second.num_bytes);
    }

    file.close();
    return true;
}

bool XArrayContainer::load(const std::string& path) 
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // 读取版本号
    uint32_t version = 0;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        throw std::runtime_error("Unsupported file version");
    }

    // 读取数组数量
    uint32_t num_arrays = 0;
    file.read(reinterpret_cast<char*>(&num_arrays), sizeof(uint32_t));

    for (uint32_t i = 0; i < num_arrays; ++i) {
        // 读取名字长度
        uint32_t key_len = 0;
        file.read(reinterpret_cast<char*>(&key_len), sizeof(uint32_t));
        
        // 读取名字字符串
        std::string key(key_len, '\0');
        file.read(&key[0], key_len);

        // 设置数据类型信息
        uint32_t dtype_code;
        file.read(reinterpret_cast<char*>(&dtype_code), sizeof(dtype_code));
        DataType type = static_cast<DataType>(dtype_code);

        // 检查类型映射是否存在
        auto it = DataTypeMap.find(dtype_code);
        if (it == DataTypeMap.end()) {
            throw std::runtime_error("Unknown data type code: " + std::to_string(dtype_code));
        }

        // 读取维度数
        uint32_t dimensions;
        file.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));
        std::vector<unsigned int> shape(dimensions);
        
        // 读取每个维度的大小
        shape.resize(dimensions);
        size_t total_elements = 1;

        for (uint32_t i = 0; i < dimensions; ++i) {
            uint32_t dim_size;
            file.read(reinterpret_cast<char*>(&dim_size), sizeof(dim_size));
            shape[i] = static_cast<unsigned int>(dim_size);
            total_elements *= dim_size;
        }

        // 设置类型信息
        const DataTypeInfo& type_info = it->second;

        // 分配新内存
        unsigned int num_bytes = static_cast<unsigned int>(total_elements * type_info.size);
        void* data = malloc(num_bytes);
        if (!data) {
            throw std::bad_alloc();
        }

        // 读取数据
        file.read(static_cast<char*>(data), num_bytes);

        if (file.gcount() != static_cast<std::streamsize>(num_bytes)) {
            throw std::runtime_error("Unknown exception in file.");
        }

        XArray& array = array_map[key] = XArray();
        array.initialize(shape, type_info.type, data, false);
        assert(array.num_bytes == num_bytes);
    }

    file.close();
    return true;
}

const XArray& XArrayContainer::operator[](const std::string& key) const
{
    if (array_map.find(key) != array_map.end()) {
        return array_map.at(key);
    }
    else {
        throw std::runtime_error("Key not found: " + key);
    }
}