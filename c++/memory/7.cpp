
#include <iostream>
using namespace std;
 
class A {
public:
	int m_k;
	int m_t;
	A() {
		cout << "construct...." << endl;
	}
	~A() {
		cout << "destruct...." << endl;
	}
	A(A &a) {
		cout << "copy construct..." << endl;
	}
};
 
//  如果函数返回值是一个对象，要考虑return语句的效率
A getObj() {
	/***********  直接返回临时对象  *************/
	// 编译器直接把临时对象创建并初始化在外部
	// 存储单元(主调函数的栈帧上)中，省去了拷
	// 贝和析构的花费，提高了效率
	/*****************************************/
	return A();  			
}
int main() {
	getObj();   //  外部存储单元
	return 0;
}