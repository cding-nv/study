#include <iostream>
#include <typeinfo>

class A {
public:
    A() {}
    virtual void foo() {
        std::cout << "foo A" << std::endl;
    }
};

class B : public A {

};

int main() {
    //A a;
    B b;
    A& a_rb = b;
    std::cout << "typeid(b).name = " << typeid(b).name() << std::endl;
    std::cout << "typeid(decltype).name = " << typeid(decltype(a_rb)).name() << std::endl;
    std::cout << "typeid(a_rb).name = " << typeid(a_rb).name() << std::endl;
    a_rb.foo();
    std::cout << "sizeof(b) = " << sizeof(b) << std::endl;
    return 0;
}
