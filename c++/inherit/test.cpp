//对于父类函数（virtual、非virtual），如果有同型函数：

//----非virtual函数由指针类型决定调用哪个

//----virtual函数由指针指向的对象决定调用哪个（运行时决定）
#include <iostream>

using namespace std;

class Base {
  public:
    void func(int i) {
        cout <<"Base::func(int i)"<< endl;
    }
    void func() {
        cout << "Base::func() " << endl;
    }
    virtual void func2(int i) {
        cout << "Base::func2(int i)" << endl;
    }
};

class Derived : public Base {
  public:
    void func(int i) {
        cout <<"Derived::func()"<< endl;
    }
    void func2(int i) {
        cout <<"Derived::func2(int i)"<< endl;
    }
};

int main() {
    Base *pb = new Derived();
    pb->func(1);  //Base::func(int i)
    pb->func();  //Base:func()
    pb->func2(1);  //Derived::func2(int i)
    delete pb;

    Derived *pd = new Derived();
    pd->func(1); //Derived::func(int i)
    // pd->func(); //不能调用
    pd->func2(1); //Derived::func2(int i)
    delete pd;
}