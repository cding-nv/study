#include <iostream>

class Shape
{
public:
    Shape(){};                    // 构造函数不能是虚函数
    virtual double calcArea() {
        std::cout << "father calcArea" << std::endl;
        return 0.0;
    }
    virtual ~Shape(){};           // 虚析构函数
};
class Circle : public Shape     // 圆形类
{
public:
    virtual double calcArea() {
        std::cout << "child calcArea" << std::endl;
        return 0.0;
    }
};
int main()
{
    Shape * shape1 = new Circle();
    shape1->calcArea();    
    delete shape1;  // 因为Shape有虚析构函数，所以delete释放内存时，先调用子类析构函数，再调用基类析构函数，防止内存泄漏。
    shape1 = NULL;
    return 0;
}