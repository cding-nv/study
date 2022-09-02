#include <iostream>

struct Test{
    Test() {
        std::cout << "Constructor" << std::endl;
    }
    Test(const Test &) {
        std::cout << "copy Constructor" << std::endl;
    }
    ~Test() {
        std::cout << "destroy" << std::endl;
    }
};

Test fun()
{
    Test t1;
    std::cout << "&t1 is " << &t1 << std::endl;
    return t1;
}

int main() {
    //Test t2 = fun();
    fun();
    //std::cout << "&t2 is " << &t2 << std::endl;
    std::cout << "This is a test!" << std::endl;
    return 0;
}
