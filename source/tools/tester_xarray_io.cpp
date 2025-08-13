
#include <stdexcept>
#include <string>
#include "tools/xarray.h"
#include "tools/xarray_template.h"


#if 0
void benchmark1()
{
    //XArray arr;
    //arr.shape = { 3, 3 };
    //arr.data = {
    //    1.123456f, 2.234567f, 3.345678f,
    //    4.456789f, 5.567890f, 6.678901f,
    //    7.789012f, 8.890123f, 9.901234f
    //};
    //arr.print();  // �������ʾ6λС��

    // ��ȡ1D����
    XArray data_1d;
    data_1d.load("X:\\project\\xhuman\\cache\\data_1d.bin");
    data_1d.print();

    // ��ȡ2D����
    XArray data_2d;
    data_2d.load("X:\\project\\xhuman\\cache\\data_2d.bin");
    data_2d.print();

    // ��ȡ3D����
    XArray data_3d;
    data_3d.load("X:\\project\\xhuman\\cache\\data_3d.bin");
    data_3d.print();
}

int benchmark2() 
{
    XArrayContainer container;
    container.load("X:\\project\\xhuman\\cache\\data.bin");

    return 0;
}


int main(int argc, char** argv)
{
    //benchmark1();
    //benchmark2();

    //XArrayContainer container;
    //container.load("X:\\checkpoints\\deep-3d-reconstruction\\cpp\\face_masking.bin");
    //container.printAll();


    // ����int32���͵����� (���ʹ���41)
    XArrayInt32 int32Array;


    return 0;


#ifdef _MSC_VER
    system("pause");
#endif
    return 0;
}
#endif