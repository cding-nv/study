
#include <iostream>
using namespace std;
 
class A {
public:
	int m_k;
	int m_t;
	A(int k, int t) :m_k(k), m_t(t) {
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
        /************** 返回本地对象 ****************/	
              /* 以下这种写法实际上执行了三步:
                  1. 构造本地对象a
                  2. 调用拷贝构造，将本地对象a拷贝到外部存储器
                  3. 调用析构函数析构本地对象a
              */
         /******************************************/
	A a(3, 4);
	return a;
}
int main() {
	getObj();   //  外部存储单元
	return 0;
}