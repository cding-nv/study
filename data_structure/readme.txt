https://github.com/NearXdu/huawei/blob/master/blog/blog.md

1. 

2. 
alloc �� malloc ������  
     alloc ����ָ��  �ҷ�����ջ��   malloc���ÿ⺯�� �����ڶ���
inline��һ�㺯��������
inline ���Ե���  ��define����
c++��Ĭ�϶�����ĸ�����   ���� ���� copy ��ֵ
ָ������õ�����    �ڳ�ʼ�����޸���

3.  
3.1 ����
��������:https://juejin.im/post/5d507199e51d4561cc25f00c
����: https://www.cnblogs.com/chengxiao/p/6129630.html
        https://www.cnblogs.com/onepixel/p/7674659.html

  3.2 .  ��̬�滮

     [2]
    [3, 4]
   [6, 5, 7]
 [4, 1, 8, 3]

class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int row = triangle.size() - 1;        
        for (int i = row - 1; i >=0; i--) {
            int col = triangle[i].size();
            for (int j = 0; j < col; j++) {
                triangle[row][j] = min(triangle[row][j], triangle[row][j+1]) + triangle[i][j];
            }
        }
        return triangle[row][0];
    }
};
a. �������ε����ڶ��п�ʼ��ȡ���ߵĽ�СֵȻ���ټ��ϵ�ǰԪ�أ����ɵó���ĳһλ�õ����±ߵ���С·����
   �Դ����ƣ��Ϳ���������ϱ�Ԫ�ص���С·����
   min(4,1) + 6,  min(1,8) + 5, min(8,3) + 7   -> 7, 6, 10, 3
   min(7,6) + 3, min(6, 10) + 4                -> 9, 10, 10, 3
   min(9, 10) + 2                              -> 11  ��Ϊ����
b. ��ԭ��������һ�н��м��㼴�ɣ������½��ڴ�

3.3  ̰��
/home/cding/c++/tests/greedy
������ڶ��������Χ�ڿ�ȷ����С https://blog.csdn.net/sarkuya/article/details/6554538  
����ڽ�����������ĺ����з�������ĸ���Ԫ�أ����ڶ����������Χ�������С��Ϊ��һ������������  
void testArrayArg2(int a[], int arrayLength)
��Ʊ������� https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/solution/mai-mai-gu-piao-de-zui-jia-shi-ji-ii-by-leetcode/
����վ: https://leetcode-cn.com/problems/gas-station/solution/jia-you-zhan-by-leetcode/
���ǹ�  https://leetcode-cn.com/problems/candy/solution/fen-fa-tang-guo-by-leetcode/

5. 
vector      https://blog.csdn.net/msdnwolaile/article/details/52708144
       push_back
       size() 
        at(i)
for_each
lambda
decltype
sort
map   
     block.insert(std::pair<std::string, std::string>(key, value))
     block.size()
     block.clear()
     block.insert() 
     block.at("type")
     block.find("height")
      block.end()
          
erase
string
shared_ptr  unique_ptr weak_ptr  ʹ��ʱ�����ͷ�ļ�<memory>

6. 
��ֵ����   https://blog.csdn.net/xiaolewennofollow/article/details/52559306
��ֵ����������֧��ת������ġ�ת��������Խ���Դ ( �ѣ�ϵͳ����� ) ��һ������ת�Ƶ���һ�����������ܹ����ٲ���Ҫ����ʱ����Ĵ����������Լ����٣��ܹ��������� C++ Ӧ�ó�������ܡ���ʱ�����ά�� ( ���������� ) ������������Ӱ��

7. 
������    ȫ��������   ����   ջ��
static 
    ��̬��Ա���������Է������еķǾ�̬��Ա�����ͷǾ�̬��Ա����, ��Ϊ��̬��Ա�������������ʱû�з���thisָ��. ����취����̬��Ա��������һ�����ָ���������Ϊ����
    ��̬����ֻ�������������ļ����пɼ������ܱ������ļ�ʹ��, �����ļ��п��Զ�����ͬ���ֵĺ��������ᷢ����ͻ
     ��̬ȫ�ֱ��� ��ȫ�������������ڴ�, δ����ʼ���ľ�̬ȫ�ֱ����ᱻ�����Զ���ʼ��Ϊ0, ��̬ȫ�ֱ������������������ļ����ǿɼ��ģ������ļ�֮���ǲ��ɼ���
     ��̬�ֲ����� Ҳ��ȫ�������������ڴ�, ʼ��פ����ȫ����������ֱ���������н�����������Ϊ�ֲ��������ں�����������
     ���еľ�̬���ݳ�Ա��������ĳ�Ա, ֻ����һ���ڴ棬�����ж�����, ������Զ������, ��û�в��������ʱҲ���Բ���. 
    ͬȫ�ֱ������, ��̬���ݳ�Աû�н�������ȫ�����ֿռ䣬��˲����������������ȫ�����ֳ�ͻ�Ŀ�����, ����ʵ����Ϣ���ء���̬���ݳ�Ա������private��Ա����ȫ�ֱ�������. ȫ�ֱ�������ȫ��������, �������ļ��ɼ�, ��Ҫextern.
    static ȫ�ֱ���:�ı����÷�Χ�����ı�洢λ��
    static �ֲ��������ı�洢λ�ã����ı����÷�Χ

���Ա�����̳�(virtual, ��virtual)
    ���ڸ��ຯ����virtual����virtual�������������ͬ�ͺ�����
    1. ��virtual������ָ�����;��������ĸ�, ����Ǹ���ָ��͵�����ĺ���, ����������ָ��͵�����ĺ���
    2. virtual������ָ��ָ��Ķ�����������ĸ�������ʱ������
    ///home/cding/c++/inherit

8.
�������� �������麯����?
    C++���м̳�ʱ��������������Ϊ�麯������������麯������ʹ��ʱ���ܴ�������й©������. �����ø���ָ��ָ�򴴽����������, delete ���ָ��ʱ,
     1. �������������麯����deleteʱ��������඼�ᱻ�ͷţ�
     2. ���������������麯����deleteʱֻ�ͷŸ��࣬���ͷ����ࣻ

9
const
c++�����Ա�������const: ��ʾ��Ա�������������thisָ��Ϊconstָ��,���ı������ݳ�Ա. һ���޸����ݳ�Ա, ��������������, mutable���εı�������. const��Ա�������ܵ��÷�const��Ա��������Ϊ��const��Ա�������Ի��޸ĳ�Ա����
c++ǰ��ʹ��const ��ʾ����ֵΪconst
const���γ�Ա����:
    1. const���γ���ֻ�ܳ�ʼ��һ��,�ٴθ�ֵ����
    2. constλ��ָ������ʾָ����ָ�����ǳ���
    3. constλ��ָ���Ҳ��ʾָ�뱾���ǳ���, ����ָ�������ڴ��ַ, ��ַ�ϵ����ݿ���ͨ�������޸�
const���κ������� ��ʾ�����ں����ڲ����Ըı�
    
using namespace
�����������ֿռ�Ҳ�Ƿ�ֹ�����ظ������ã����־ֲ���, ֻ��û�����ֶ���, ����static

static ��ȫ�ֱ���������
    C++ ȫ�ֱ������ֲ���������̬ȫ�ֱ�������̬�ֲ�����������

���ֿռ� using  û�����ֵ� ...
 
10
// �ǵݹ�������
public static void postorderTraversal(TreeNode root) {
    Stack<TreeNode> treeNodeStack = new Stack<TreeNode>();
    TreeNode node = root;
    TreeNode lastVisit = root;
    while (node != null || !treeNodeStack.isEmpty()) {
        while (node != null) {
            treeNodeStack.push(node);
            node = node.left;
        }
        //�鿴��ǰջ��Ԫ��
        node = treeNodeStack.peek();
        //�����������ҲΪ�գ������������Ѿ�����
        //�����ֱ�������ǰ�ڵ��ֵ
        if (node.right == null || node.right == lastVisit) {
            System.out.print(node.val + " ");
            treeNodeStack.pop();
            lastVisit = node;
            node = null;
        } else {
            //���򣬼�������������
            node = node.right;
        }
    }
}


10.19.225.118:  /home/cding/c++
\\10.19.225.118\cding\c++\sort_lambda

-3.  double log (double);  ��eΪ�׵Ķ���
     double log10 (double); ��10Ϊ�׵Ķ���
     double pow(double x,double y); ����x��y����
     float powf(float x,float y); ������powһ�£�ֻ�������������Ϊ�����ȸ�����
    double exp (double); ��ȡ��Ȼ��e����
    double sqrt (double); ��ƽ����
    #include <math.h> ���������
    �����log(a)b�Ļ�����ѧ���� f = log(b) / log(a);

-2.
    std::vector<std::map<std::string, std::string>> blocks;
    std::map<std::string, std::string> block;
    block.insert(std::pair<std::string, std::string>(key, value));

    std::stoul()
    std::stoi()
    std::stof()
    std::string.find_first_of(',')
    std::string.erase(0, npos+1)

-1. basic_string::npos
static const size_type npos = -1;//����
The constant is the largest representable value of type size_type. It is assuredly larger than max_size(); hence it serves as either a very large value or as a special code.
���ϵ���˼��npos��һ����������ʾsize_t�����ֵ��Maximum value for size_t��������������ṩ���������������ʾ�����ڵ�λ�ã�����һ����std::container_type::size_type��

0. decltype
https://www.cnblogs.com/QG-whz/p/4952980.html
decltype��auto�ؼ���һ�������ڽ��б���ʱ�����Ƶ�����������auto������һЩ����ġ�decltype�������Ƶ���������autoһ���Ǵӱ��������ĳ�ʼ�����ʽ��ñ��������ͣ�����������һ����ͨ���ʽ��Ϊ���������ظñ��ʽ������,����decltype������Ա��ʽ������ֵ��

C++��ֹ�౻�̳� http://www.voidcn.com/article/p-meaqrscj-he.html
https://www.cnblogs.com/xkfz007/archive/2012/07/21/2602110.html
https://zhuanlan.zhihu.com/p/41309205

https://github.com/huihut/interview

1. ���������ʼ�� ����ɾ�� https://blog.csdn.net/m_zhurunfeng/article/details/54809821
typedef int ElemType;
typedef struct Node {
    ElemType data;
    struct Node *next;
} Node, *linkedList;
��ʼ����
 linkedList init() {
     Node *L;
    L = (Node*) malloc (sizeof(Node));
    if (L == NULL) {
        printf
    }  
    L->next = NULL;
    return L;
} 
ͷ����
    Node* p = malloc����
    p-> data = 
    p->next = L-> next;
    L->next = p;
β����
    Node* p = malloc����
    p-> data = 
    r->next = p;
    r = p;
    r-> next = NULL
���� λ��p
    for (i=0; i< p; i ++) {
       pre = pre->next;
    }
    Node* p = malloc()
    p->data = 
    p->next = pre->next;
    pre->next = p;

2. stack    typedef struct {
    ElemType * elem;   ����ַ
     int top;        
     int size;
     int increment;
}

3. ����
  typedef struct {
      ElemType * elem;
     int front;
     int rear;
    int maxSize;
  }




4. ������




5 ���̣���Դ����Ķ�����λ
�̣߳�schedual ���ȵĶ�����λ
����ͨ�ţ� �ܵ� ��Ϣ���У� �����ڴ棬 �׽��֣� �źţ� �ź����� netlink
�߳�ͨ�ţ��������� ��ѡ���� ��������
�̼߳��ͨ��Ŀ����Ҫ�������߳�ͬ���������߳�û�������ͨ���е��������ݽ�����ͨ�Ż���
����֮��˽�к͹������Դ
˽�У���ַ�ռ䡢�ѡ�ȫ�ֱ�����ջ���Ĵ���
��������Σ��������ݣ�����Ŀ¼������ ID
�߳�֮��˽�к͹������Դ
˽�У��߳�ջ���Ĵ��������������
�����ѣ���ַ�ռ䣬ȫ�ֱ�������̬����

6. TCP ��������ȫ���̽���
�ͻ��˷��� SYN ����������˵���ͻ������������ӣ�
������յ��ͻ��˷��� SYN�����ظ� SYN+ACK ���ͻ��ˣ�ͬ�⽨�����ӣ���
�ͻ����յ�����˵� SYN+ACK �󣬻ظ� ACK ������ˣ���ʾ�ͻ����յ��˷���˷���ͬ�ⱨ�ģ���
������յ��ͻ��˵� ACK�������ѽ������������ݴ��䡣
TCP ΪʲôҪ�����������֣�
����һ����Ϊ�ŵ����ɿ����� TCP ���ڲ��ɿ��ŵ��Ͻ����ɿ��ش��䣬��ô����ͨ���������ϵ���Сֵ������ UDP ���轨���ɿ����䣬��� UDP ����Ҫ�������֡���

�鲢����  https://www.cnblogs.com/chengxiao/p/6194356.html 
����
�����㷨 ƽ��ʱ�临�Ӷ� ���ʱ�临�Ӷ� �ռ临�Ӷ� ���ݶ����ȶ���
ð������ O(n2) O(n2) O(1) �ȶ�
ѡ������ O(n2) O(n2) O(1) ���鲻�ȶ��������ȶ�
�������� O(n2) O(n2) O(1) �ȶ�
�������� O(n*log2n) O(n2) O(log2n) ���ȶ�
������ O(n*log2n) O(n*log2n) O(1) ���ȶ�
�鲢���� O(n*log2n) O(n*log2n) O(n) �ȶ�
ϣ������ O(n*log2n) O(n2) O(1) ���ȶ�
�������� O(n+m) O(n+m) O(n+m) �ȶ�
Ͱ���� O(n) O(n) O(m) �ȶ�
�������� O(k*n) O(n2)  �ȶ�


��װ
�ѿ͹������װ�ɳ�����࣬��������԰��Լ������ݺͷ���ֻ�ÿ��ŵ�����߶���������Բ����ŵĽ�����Ϣ���ء��ؼ��֣�public, protected, private����дĬ��Ϊ private��
public ��Ա�����Ա�����ʵ�����
protected ��Ա��ֻ�������༰����ĳ�Ա��������
private ��Ա��ֻ��������ĳ�Ա��������Ԫ�����Ԫ��������
�̳�
���ࣨ���ࣩ����> �����ࣨ���ࣩ
��̬
��̬��������״̬����̬��������˵�����ǿ��Խ���̬����Ϊ��Ϣ�Զ�����ʽ��ʾ��������
��̬���Է�װ�ͼ̳�Ϊ�����ġ�
C++ ��̬���༰ʵ�֣�
                            i.          ���ض�̬��Ad-hoc Polymorphism�������ڣ����������ء����������
                           ii.          �����Ͷ�̬��Subtype Polymorphism�������ڣ����麯��
                          iii.          ������̬�ԣ�Parametric Polymorphism�������ڣ�����ģ�塢����ģ��
                          iv.          ǿ�ƶ�̬��Coercion Polymorphism��������/�����ڣ�����������ת�����Զ�������ת��
 
��̬��̬��������/��󶨣�
��������
class A{
public:
    void do(int a);
    void do(int a, int b);
};
��̬��̬����������/��󶨣�
�麯������ virtual ���γ�Ա������ʹ���Ϊ�麯��
��ͨ�����������Ա�������������麯��
��̬������static���������麯��
���캯���������麯������Ϊ�ڵ��ù��캯��ʱ�����ָ�벢û���ڶ�����ڴ�ռ��У�����Ҫ���캯��������ɺ�Ż��γ����ָ�룩
�������������Ǳ��ֶ�̬��ʱ���麯�������ͼ����麯����virtual������������������inline����
 
�麯����:https://www.jianshu.com/p/64f3b9c22898  
 
�麯����virtual������������������inline����
Are "inline virtual" member functions ever actually "inlined"?
�麯�����������������������ǿ��������麯���ģ����ǵ��麯�����ֶ�̬�Ե�ʱ����������
�������ڱ�����������������������麯���Ķ�̬���������ڣ��������޷�֪�������ڵ����ĸ����룬����麯������Ϊ��̬��ʱ�������ڣ�������������
inline virtual Ψһ����������ʱ���ǣ�������֪�������õĶ������ĸ��ࣨ�� Base::who()������ֻ���ڱ���������ʵ�ʶ�������Ƕ����ָ�������ʱ�Żᷢ��

1. How to make a class not be derived.
class Uninheritable {
    friend class NotABase;
private:
    Uninheritable(void) {}
};
 
class NotABase: public virtual Uninheritable {
    // WHATEVER
};
 
class NotADerived: public NotABase {
    // WHATEVER ELSE
};
a. NotABase is Uninheritable��s friend, so NotABase can call Uninheritable��s private constructor.
b. NotABase : public virtual Uninheritable, so NotADerived and NotABase share the same virtual base,
    but NotADerived can��t call  Uninheritable��s private constructor.
 
2. What happen when function returns object.
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
    fun();
 
    std::cout << "This is a test!" << std::endl;
    return 0;
}
 
    a.
$ g++ -o test ./test.cpp -fno-elide-constructors
$ ./test
Constructor
&t1 is 0x7ffcd1e93257
copy Constructor
destroy
destroy
This is a test!
 
        Call func() -> t1 constructor -> in return, call copy constructor to generate a temporary object -> t1 destroy -> temporary object destroy.
    b.
$ g++ -o test1 ./test.cpp
$ ./test1
Constructor
&t1 is 0x7ffdf8f1be67
destroy
This is a test!
     Without ��-fno-elide-constructors��, g++ compile will do RVO ��Return Value Optimization��, don��t generate temporary object. https://www.cnblogs.com/xkfz007/archive/2012/07/21/2602110.html
