#include <iostream>

class Shape
{
public:
    Shape(){};                    // ���캯���������麯��
    virtual double calcArea() {
        std::cout << "father calcArea" << std::endl;
        return 0.0;
    }
    virtual ~Shape(){};           // ����������
};
class Circle : public Shape     // Բ����
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
    delete shape1;  // ��ΪShape������������������delete�ͷ��ڴ�ʱ���ȵ������������������ٵ��û���������������ֹ�ڴ�й©��
    shape1 = NULL;
    return 0;
}