
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
 
//  �����������ֵ��һ������Ҫ����return����Ч��
A getObj() {
        /************** ���ر��ض��� ****************/	
              /* ��������д��ʵ����ִ��������:
                  1. ���챾�ض���a
                  2. ���ÿ������죬�����ض���a�������ⲿ�洢��
                  3. �������������������ض���a
              */
         /******************************************/
	A a(3, 4);
	return a;
}
int main() {
	getObj();   //  �ⲿ�洢��Ԫ
	return 0;
}