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


//Test fun()
//{
//    Test t1;
//    std::cout << "&t1 is " << &t1 << std::endl;
//    return t1;
//}

Test func2(Test &t3) {
    std::cout << "&t3 is " << &t3 << std::endl;
    return t3;
}

int main() {
    Test t3;
    Test t2 = func2(t3);
    std::cout << "&t2 is " << &t2 << std::endl;
    std::cout << "This is a test!" << std::endl;
    return 0;
}
