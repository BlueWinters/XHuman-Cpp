
#ifndef __XArray__
#define __XArray__

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <typeinfo>
#include "xarray_dtype.h"


class XArray
{
public:
    DataTypeInfo type_info;
    std::vector<unsigned int> shape;
    unsigned int num_bytes;
    void* data;

public:
    // 默认构造函数
    XArray();
    // 带参数的构造函数
    XArray(const std::vector<unsigned int>& shape, DataTypeCode data_type_code, void* data, bool copy_data = true);
    // 析构函数
    ~XArray();

public:
    // 清理数据
    void clear();
    // 初始化数据
    void initialize(const std::vector<unsigned int>& shape, DataTypeCode data_type_code, void* data, bool copy_data = false);
    // 获取维度数
    int dimensions() const;
    // 获取总的元素数量
    size_t size() const;
    // 获取元素大小
    size_t getElementSize() const;
    // 获取元素个数
    size_t getNumElements() const;
    // 打印单个元素
    void printElement(size_t index) const;
    // 输出基本信息
    void printInfo() const;
    // 输出数组信息
    void print() const;
    // 从文件读取
    bool load(const std::string& path);

protected:
    // 输出数组信息(一维)
    void print1D() const;
    // 输出数组信息(二维)
    void print2D() const;
    // 输出数组信息(三维)
    void print3D() const;
};


// 顶层容器：键值对数组存储
class XArrayContainer
{
public:
    XArrayContainer() = default;
    ~XArrayContainer() = default;
public:
    std::map<std::string, XArray> array_map;

public:
    // 添加数组
    void addArray(const std::string& key, const XArray& array);
    // 获取数组
    bool getArray(const std::string& key, XArray& out_array) const;
    // 删除数组
    void removeArray(const std::string& key);
    // 检查是否存在
    bool hasArray(const std::string& key) const;
    // 获取所有key列表
    std::vector<std::string> keys() const;
    // 输出所有数组信息
    void printAll() const;
    // 保存到文件
    bool save(const std::string& path) const;
    // 从文件加载
    bool load(const std::string& path);

public:
    const XArray& operator[](const std::string& key) const;
};


#endif