#include <iostream>

struct A {
    int ax;
    virtual void f0() {}
    virtual void f1() {}
};

struct B : public A {
    int bx;
    void f0() override {};
};

int main() {
    A a;
    B b;
    std::cout << "sizeof(a) " << sizeof(a) << std::endl;
    std::cout << "sizeof(b) " << sizeof(b) << std::endl;
    return 0;
}
