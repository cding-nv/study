#include <iostream>
#include <typeinfo>

class A {
public:
    A() {}
    void foo() {
        std::cout << "foo A" << std::endl;
    }
};

int main() {
    A a;
    std::cout << "typeid(a).name = " << typeid(a).name() << std::endl;
    a.foo();
    std::cout << "sizeof(a) = " << sizeof(a) << std::endl;
    return 0;
}
