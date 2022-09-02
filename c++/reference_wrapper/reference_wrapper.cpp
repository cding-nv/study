#include <iostream>
#include <functional>
using namespace std;

void func(int a, int b)
{
    cout << "a = " << a <<",";
    cout << "b = " << b << endl;
}

void func1(int i)
{
    cout << "i = " << i << endl;
}

int main()
{
    // 包裹函数指针
    int x=5, y=7;
    std::reference_wrapper<void(int,int)> f0 = func;
    f0(5,7);

    auto f1 = std::ref(func);
    f1(5,7);

    // 和bind结合使用
    int i = 10;

    auto f2 = std::bind(func1, i);
    auto f3 = std::bind(func1, std::ref(i));

    i = 30;

    f2();
    f3();

    return 0;
}