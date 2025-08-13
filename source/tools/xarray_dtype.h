
#ifndef __XArray_DataType__
#define __XArray_DataType__

#include <map>
#include <string>
#include <typeinfo>


// ��������ö��
enum class DataTypeCode : uint32_t
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

// ����������Ϣ�ṹ
class DataTypeInfo 
{
public:
    DataTypeCode type;
    size_t size;
    std::string name;
public:
    // ���캯��
    DataTypeInfo(DataTypeCode t, size_t s, const std::string& n)
        : type(t), size(s), name(n) {}
    DataTypeInfo() : type(DataTypeCode::NONE), size(0), name("none") {}
    DataTypeInfo(const DataTypeInfo& other)
        : type(other.type), size(other.size), name(other.name) {}
public:
    // �Ⱥ����ط�
    DataTypeInfo& operator=(const DataTypeInfo& other) {
        if (this != &other) {  // �Ը�ֵ���
            type = other.type;
            size = other.size;
            name = other.name;
        }
        return *this;
    }
};


// ��������ӳ���
const std::map<int, DataTypeInfo> DataTypeMap = {
    {0, DataTypeInfo(DataTypeCode::NONE, 0, "none")},
    {11, DataTypeInfo(DataTypeCode::BYTE, 1, "byte")},
    {12, DataTypeInfo(DataTypeCode::INT8, 1, "int8")},
    {13, DataTypeInfo(DataTypeCode::UINT8, 1, "uint8")},
    {21, DataTypeInfo(DataTypeCode::INT16, 2, "int16")},
    {22, DataTypeInfo(DataTypeCode::UINT16, 2, "uint16")},
    {23, DataTypeInfo(DataTypeCode::FLOAT16, 2, "float16")},
    {41, DataTypeInfo(DataTypeCode::INT32, 4, "int32")},
    {42, DataTypeInfo(DataTypeCode::UINT32, 4, "uint32")},
    {43, DataTypeInfo(DataTypeCode::FLOAT32, 4, "float32")},
    {82, DataTypeInfo(DataTypeCode::INT64, 8, "int64")},
    {83, DataTypeInfo(DataTypeCode::FLOAT64, 8, "float64")}
};



#endif