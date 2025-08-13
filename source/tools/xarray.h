
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
    // Ĭ�Ϲ��캯��
    XArray();
    // �������Ĺ��캯��
    XArray(const std::vector<unsigned int>& shape, DataTypeCode data_type_code, void* data, bool copy_data = true);
    // ��������
    ~XArray();

public:
    // ��������
    void clear();
    // ��ʼ������
    void initialize(const std::vector<unsigned int>& shape, DataTypeCode data_type_code, void* data, bool copy_data = false);
    // ��ȡά����
    int dimensions() const;
    // ��ȡ�ܵ�Ԫ������
    size_t size() const;
    // ��ȡԪ�ش�С
    size_t getElementSize() const;
    // ��ȡԪ�ظ���
    size_t getNumElements() const;
    // ��ӡ����Ԫ��
    void printElement(size_t index) const;
    // ���������Ϣ
    void printInfo() const;
    // ���������Ϣ
    void print() const;
    // ���ļ���ȡ
    bool load(const std::string& path);

protected:
    // ���������Ϣ(һά)
    void print1D() const;
    // ���������Ϣ(��ά)
    void print2D() const;
    // ���������Ϣ(��ά)
    void print3D() const;
};


// ������������ֵ������洢
class XArrayContainer
{
public:
    XArrayContainer() = default;
    ~XArrayContainer() = default;
public:
    std::map<std::string, XArray> array_map;

public:
    // �������
    void addArray(const std::string& key, const XArray& array);
    // ��ȡ����
    bool getArray(const std::string& key, XArray& out_array) const;
    // ɾ������
    void removeArray(const std::string& key);
    // ����Ƿ����
    bool hasArray(const std::string& key) const;
    // ��ȡ����key�б�
    std::vector<std::string> keys() const;
    // �������������Ϣ
    void printAll() const;
    // ���浽�ļ�
    bool save(const std::string& path) const;
    // ���ļ�����
    bool load(const std::string& path);

public:
    const XArray& operator[](const std::string& key) const;
};


#endif