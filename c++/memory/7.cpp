
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
 
//  �����������ֵ��һ������Ҫ����return����Ч��
A getObj() {
	/***********  ֱ�ӷ�����ʱ����  *************/
	// ������ֱ�Ӱ���ʱ���󴴽�����ʼ�����ⲿ
	// �洢��Ԫ(����������ջ֡��)�У�ʡȥ�˿�
	// ���������Ļ��ѣ������Ч��
	/*****************************************/
	return A();  			
}
int main() {
	getObj();   //  �ⲿ�洢��Ԫ
	return 0;
}