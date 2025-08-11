
#ifndef __XArray_DataType__
#define __XArray_DataType__

#include <map>
#include <string>
#include <typeinfo>


// 数据类型枚举
enum class DataType : uint32_t
{
    // 0-byte
    NONE = 0,
    // 1-byte
    BYTE = 11,
    INT8 = 12,
    UINT8 = 13,
    // 2-byte
    INT16 = 21,
    UINT16 = 22,
    FLOAT16 = 23,
    // 4-byte
    INT32 = 41,
    UINT32 = 42,
    FLOAT32 = 43,
    // 8-byte
    INT64 = 81,
    FLOAT64 = 82
};

// 数据类型信息结构
class DataTypeInfo 
{
public:
    DataType type;
    size_t size;
    std::string name;
public:
    // 构造函数
    DataTypeInfo(DataType t, size_t s, const std::string& n)
        : type(t), size(s), name(n) {}
    DataTypeInfo() : type(DataType::NONE), size(0), name("none") {}
    DataTypeInfo(const DataTypeInfo& other)
        : type(other.type), size(other.size), name(other.name) {}
public:
    // 等号重载符
    DataTypeInfo& operator=(const DataTypeInfo& other) {
        if (this != &other) {  // 自赋值检查
            type = other.type;
            size = other.size;
            name = other.name;
        }
        return *this;
    }
};


// 数据类型映射表
const std::map<int, DataTypeInfo> DataTypeMap = {
    {0, DataTypeInfo(DataType::NONE, 0, "none")},
    {11, DataTypeInfo(DataType::BYTE, 1, "byte")},
    {12, DataTypeInfo(DataType::INT8, 1, "int8")},
    {13, DataTypeInfo(DataType::UINT8, 1, "uint8")},
    {21, DataTypeInfo(DataType::INT16, 2, "int16")},
    {22, DataTypeInfo(DataType::UINT16, 2, "uint16")},
    {23, DataTypeInfo(DataType::FLOAT16, 2, "float16")},
    {41, DataTypeInfo(DataType::INT32, 4, "int32")},
    {42, DataTypeInfo(DataType::UINT32, 4, "uint32")},
    {43, DataTypeInfo(DataType::FLOAT32, 4, "float32")},
    {82, DataTypeInfo(DataType::INT64, 8, "int64")},
    {83, DataTypeInfo(DataType::FLOAT64, 8, "float64")}
};



#endif