https://github.com/NearXdu/huawei/blob/master/blog/blog.md

1. 

2. 
alloc 和 malloc 的区别  
     alloc 会变成指令  且分配在栈上   malloc调用库函数 分配在堆上
inline和一般函数的区别
inline 可以调试  但define不行
c++类默认定义的四个函数   构造 析构 copy 赋值
指针和引用的区别    在初始化及修改上

3.  
3.1 排序
快速排序:https://juejin.im/post/5d507199e51d4561cc25f00c
排序: https://www.cnblogs.com/chengxiao/p/6129630.html
        https://www.cnblogs.com/onepixel/p/7674659.html

  3.2 .  动态规划

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
a. 从三角形倒数第二行开始，取两者的较小值然后再加上当前元素，即可得出从某一位置到最下边的最小路径和
   以此类推，就可以求出最上边元素的最小路径和
   min(4,1) + 6,  min(1,8) + 5, min(8,3) + 7   -> 7, 6, 10, 3
   min(7,6) + 3, min(6, 10) + 4                -> 9, 10, 10, 3
   min(9, 10) + 2                              -> 11  即为所求
b. 在原数组的最后一行进行计算即可，不用新建内存

3.3  贪心
/home/cding/c++/tests/greedy
数组仅在定义其的域范围内可确定大小 https://blog.csdn.net/sarkuya/article/details/6554538  
如果在接受数组参数的函数中访问数组的各个元素，需在定义数组的域范围将数组大小作为另一辅助参数传递  
void testArrayArg2(int a[], int arrayLength)
股票收益最大化 https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/solution/mai-mai-gu-piao-de-zui-jia-shi-ji-ii-by-leetcode/
加油站: https://leetcode-cn.com/problems/gas-station/solution/jia-you-zhan-by-leetcode/
分糖果  https://leetcode-cn.com/problems/candy/solution/fen-fa-tang-guo-by-leetcode/

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
shared_ptr  unique_ptr weak_ptr  使用时需添加头文件<memory>

6. 
右值引用   https://blog.csdn.net/xiaolewennofollow/article/details/52559306
右值引用是用来支持转移语义的。转移语义可以将资源 ( 堆，系统对象等 ) 从一个对象转移到另一个对象，这样能够减少不必要的临时对象的创建、拷贝以及销毁，能够大幅度提高 C++ 应用程序的性能。临时对象的维护 ( 创建和销毁 ) 对性能有严重影响

7. 
代码区    全局数据区   堆区   栈区
static 
    静态成员函数不可以访问类中的非静态成员变量和非静态成员函数, 因为静态成员函数在类对象构造时没有分配this指针. 解决办法：静态成员函数增加一个类的指针或引用作为参数
    静态函数只能在声明它的文件当中可见，不能被其它文件使用, 其它文件中可以定义相同名字的函数，不会发生冲突
     静态全局变量 在全局数据区分配内存, 未经初始化的静态全局变量会被程序自动初始化为0, 静态全局变量在声明它的整个文件都是可见的，而在文件之外是不可见的
     静态局部变量 也在全局数据区分配内存, 始终驻留在全局数据区，直到程序运行结束。作用域为局部作用域，在函数或语句块内
     类中的静态数据成员当作是类的成员, 只分配一次内存，供所有对象共用, 对象可以对其更新, 在没有产生类对象时也可以操作. 
    同全局变量相比, 静态数据成员没有进入程序的全局名字空间，因此不存在与程序中其它全局名字冲突的可能性, 可以实现信息隐藏。静态数据成员可以是private成员，而全局变量不能. 全局变量具有全局作用域, 对所有文件可见, 需要extern.
    static 全局变量:改变作用范围，不改变存储位置
    static 局部变量：改变存储位置，不改变作用范围

类成员函数继承(virtual, 非virtual)
    对于父类函数（virtual、非virtual），如果子类有同型函数：
    1. 非virtual函数由指针类型决定调用哪个, 如果是父类指针就调父类的函数, 如果是子类的指针就调子类的函数
    2. virtual函数由指针指向的对象决定调用哪个（运行时决定）
    ///home/cding/c++/inherit

8.
析构函数 可以是虚函数吗?
    C++类有继承时，析构函数必须为虚函数。如果不是虚函数，则使用时可能存在内在泄漏的问题. 比如用父类指针指向创建的子类对象, delete 这个指针时,
     1. 若析构函数是虚函数，delete时父类和子类都会被释放；
     2. 若析构函数不是虚函数，delete时只释放父类，不释放子类；

9
const
c++在类成员函数后加const: 表示成员函数隐含传入的this指针为const指针,不改变类数据成员. 一旦修改数据成员, 编译器按错误处理, mutable修饰的变量除外. const成员函数不能调用非const成员函数，因为非const成员函数可以会修改成员变量
c++前面使用const 表示返回值为const
const修饰成员变量:
    1. const修饰常量只能初始化一次,再次赋值报错
    2. const位于指针左侧表示指针所指数据是常量
    3. const位于指针右侧表示指针本身是常量, 不能指向其他内存地址, 地址上的数据可以通过引用修改
const修饰函数参数 表示参数在函数内不可以改变
    
using namespace
不具名的名字空间也是防止对象重复定义用，保持局部性, 只是没有名字而已, 类似static

static 和全局变量的区别
    C++ 全局变量、局部变量、静态全局变量、静态局部变量的区别

名字空间 using  没有名字的 ...
 
10
// 非递归后序遍历
public static void postorderTraversal(TreeNode root) {
    Stack<TreeNode> treeNodeStack = new Stack<TreeNode>();
    TreeNode node = root;
    TreeNode lastVisit = root;
    while (node != null || !treeNodeStack.isEmpty()) {
        while (node != null) {
            treeNodeStack.push(node);
            node = node.left;
        }
        //查看当前栈顶元素
        node = treeNodeStack.peek();
        //如果其右子树也为空，或者右子树已经访问
        //则可以直接输出当前节点的值
        if (node.right == null || node.right == lastVisit) {
            System.out.print(node.val + " ");
            treeNodeStack.pop();
            lastVisit = node;
            node = null;
        } else {
            //否则，继续遍历右子树
            node = node.right;
        }
    }
}


10.19.225.118:  /home/cding/c++
\\10.19.225.118\cding\c++\sort_lambda

-3.  double log (double);  以e为底的对数
     double log10 (double); 以10为底的对数
     double pow(double x,double y); 计算x的y次幂
     float powf(float x,float y); 功能与pow一致，只是输入与输出皆为单精度浮点数
    double exp (double); 求取自然数e的幂
    double sqrt (double); 开平方根
    #include <math.h> 加载这个库
    如果求log(a)b的话就数学方法 f = log(b) / log(a);

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
static const size_type npos = -1;//定义
The constant is the largest representable value of type size_type. It is assuredly larger than max_size(); hence it serves as either a very large value or as a special code.
以上的意思是npos是一个常数，表示size_t的最大值（Maximum value for size_t）。许多容器都提供这个东西，用来表示不存在的位置，类型一般是std::container_type::size_type。

0. decltype
https://www.cnblogs.com/QG-whz/p/4952980.html
decltype与auto关键字一样，用于进行编译时类型推导，不过它与auto还是有一些区别的。decltype的类型推导并不是像auto一样是从变量声明的初始化表达式获得变量的类型，而是总是以一个普通表达式作为参数，返回该表达式的类型,而且decltype并不会对表达式进行求值。

C++防止类被继承 http://www.voidcn.com/article/p-meaqrscj-he.html
https://www.cnblogs.com/xkfz007/archive/2012/07/21/2602110.html
https://zhuanlan.zhihu.com/p/41309205

https://github.com/huihut/interview

1. 单向链表初始化 插入删除 https://blog.csdn.net/m_zhurunfeng/article/details/54809821
typedef int ElemType;
typedef struct Node {
    ElemType data;
    struct Node *next;
} Node, *linkedList;
初始化：
 linkedList init() {
     Node *L;
    L = (Node*) malloc (sizeof(Node));
    if (L == NULL) {
        printf
    }  
    L->next = NULL;
    return L;
} 
头插入
    Node* p = malloc（）
    p-> data = 
    p->next = L-> next;
    L->next = p;
尾插入
    Node* p = malloc（）
    p-> data = 
    r->next = p;
    r = p;
    r-> next = NULL
插入 位置p
    for (i=0; i< p; i ++) {
       pre = pre->next;
    }
    Node* p = malloc()
    p->data = 
    p->next = pre->next;
    pre->next = p;

2. stack    typedef struct {
    ElemType * elem;   基地址
     int top;        
     int size;
     int increment;
}

3. 队列
  typedef struct {
      ElemType * elem;
     int front;
     int rear;
    int maxSize;
  }




4. 二叉树




5 进程：资源分配的独立单位
线程：schedual 调度的独立单位
进程通信： 管道 消息队列， 共享内存， 套接字， 信号， 信号量， netlink
线程通信：互斥锁， 自选锁， 条件变量
线程间的通信目的主要是用于线程同步，所以线程没有像进程通信中的用于数据交换的通信机制
进程之间私有和共享的资源
私有：地址空间、堆、全局变量、栈、寄存器
共享：代码段，公共数据，进程目录，进程 ID
线程之间私有和共享的资源
私有：线程栈，寄存器，程序计数器
共享：堆，地址空间，全局变量，静态变量

6. TCP 建立连接全过程解释
客户端发送 SYN 给服务器，说明客户端请求建立连接；
服务端收到客户端发的 SYN，并回复 SYN+ACK 给客户端（同意建立连接）；
客户端收到服务端的 SYN+ACK 后，回复 ACK 给服务端（表示客户端收到了服务端发的同意报文）；
服务端收到客户端的 ACK，连接已建立，可以数据传输。
TCP 为什么要进行三次握手？
【答案一】因为信道不可靠，而 TCP 想在不可靠信道上建立可靠地传输，那么三次通信是理论上的最小值。（而 UDP 则不需建立可靠传输，因此 UDP 不需要三次握手。）

归并排序：  https://www.cnblogs.com/chengxiao/p/6194356.html 
排序
排序算法 平均时间复杂度 最差时间复杂度 空间复杂度 数据对象稳定性
冒泡排序 O(n2) O(n2) O(1) 稳定
选择排序 O(n2) O(n2) O(1) 数组不稳定、链表稳定
插入排序 O(n2) O(n2) O(1) 稳定
快速排序 O(n*log2n) O(n2) O(log2n) 不稳定
堆排序 O(n*log2n) O(n*log2n) O(1) 不稳定
归并排序 O(n*log2n) O(n*log2n) O(n) 稳定
希尔排序 O(n*log2n) O(n2) O(1) 不稳定
计数排序 O(n+m) O(n+m) O(n+m) 稳定
桶排序 O(n) O(n) O(m) 稳定
基数排序 O(k*n) O(n2)  稳定


封装
把客观事物封装成抽象的类，并且类可以把自己的数据和方法只让可信的类或者对象操作，对不可信的进行信息隐藏。关键字：public, protected, private。不写默认为 private。
public 成员：可以被任意实体访问
protected 成员：只允许被子类及本类的成员函数访问
private 成员：只允许被本类的成员函数、友元类或友元函数访问
继承
基类（父类）——> 派生类（子类）
多态
多态，即多种状态（形态）。简单来说，我们可以将多态定义为消息以多种形式显示的能力。
多态是以封装和继承为基础的。
C++ 多态分类及实现：
                            i.          重载多态（Ad-hoc Polymorphism，编译期）：函数重载、运算符重载
                           ii.          子类型多态（Subtype Polymorphism，运行期）：虚函数
                          iii.          参数多态性（Parametric Polymorphism，编译期）：类模板、函数模板
                          iv.          强制多态（Coercion Polymorphism，编译期/运行期）：基本类型转换、自定义类型转换
 
静态多态（编译期/早绑定）
函数重载
class A{
public:
    void do(int a);
    void do(int a, int b);
};
动态多态（运行期期/晚绑定）
虚函数：用 virtual 修饰成员函数，使其成为虚函数
普通函数（非类成员函数）不能是虚函数
静态函数（static）不能是虚函数
构造函数不能是虚函数（因为在调用构造函数时，虚表指针并没有在对象的内存空间中，必须要构造函数调用完成后才会形成虚表指针）
内联函数不能是表现多态性时的虚函数，解释见：虚函数（virtual）可以是内联函数（inline）吗？
 
虚函数表:https://www.jianshu.com/p/64f3b9c22898  
 
虚函数（virtual）可以是内联函数（inline）吗？
Are "inline virtual" member functions ever actually "inlined"?
虚函数可以是内联函数，内联是可以修饰虚函数的，但是当虚函数表现多态性的时候不能内联。
内联是在编译器建议编译器内联，而虚函数的多态性在运行期，编译器无法知道运行期调用哪个代码，因此虚函数表现为多态性时（运行期）不可以内联。
inline virtual 唯一可以内联的时候是：编译器知道所调用的对象是哪个类（如 Base::who()），这只有在编译器具有实际对象而不是对象的指针或引用时才会发生

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
a. NotABase is Uninheritable’s friend, so NotABase can call Uninheritable’s private constructor.
b. NotABase : public virtual Uninheritable, so NotADerived and NotABase share the same virtual base,
    but NotADerived can’t call  Uninheritable’s private constructor.
 
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
     Without “-fno-elide-constructors”, g++ compile will do RVO “Return Value Optimization”, don’t generate temporary object. https://www.cnblogs.com/xkfz007/archive/2012/07/21/2602110.html
