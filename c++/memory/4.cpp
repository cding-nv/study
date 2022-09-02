#include <iostream>

class A {
public:
    virtual void f0() {
        std::cout << "A\n";
    }
    A() {
        this->f0();
    }
    virtual ~A() {
        this->f0();
    }
};

class B : public A {
public:
    virtual void f0() {
        std::cout << "B\n";
    }
    B() {
        this->f0();
    }
    ~B() override {
        this->f0();
    }
};

int main() {
    B b;
    return 0;
}
