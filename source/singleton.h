
#ifndef __Singleton__
#define __Singleton__


// 线程安全单例宏 - 默认构造函数和析构函数
#define THREAD_SAFE_SINGLETON_DEFAULT(ClassName) \
public: \
    static ClassName& getInstance() { \
        static ClassName instance; /* c++11静态局部变量，自动线程安全 */ \
        return instance; \
    } \
    /* 禁止拷贝和赋值 */ \
    ClassName(const ClassName&) = delete; \
    ClassName& operator=(const ClassName&) = delete; \
    ClassName(ClassName&&) = delete; \
    ClassName& operator=(ClassName&&) = delete; \
protected: \
    /* 受保护的构造函数和析构函数，无法自行创建对象 */ \
    ClassName() = default; \
    ~ClassName() = default


// 线程安全单例宏 - 自定义构造函数和析构函数
#define THREAD_SAFE_SINGLETON_AUTOMATIC(ClassName) \
public: \
    static ClassName& getInstance() { \
        static ClassName instance; /* c++11静态局部变量，自动线程安全 */ \
        return instance; \
    } \
    /* 禁止拷贝和赋值 */ \
    ClassName(const ClassName&) = delete; \
    ClassName& operator=(const ClassName&) = delete; \
    ClassName(ClassName&&) = delete; \
    ClassName& operator=(ClassName&&) = delete;


#endif

