#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
using namespace std;

class Foo {
  public:
    int a;
    int b;

    Foo():a(0), b(0) {}
    ~Foo() {}

    Foo(int a, int b) {
        this->a = a;
        this->b = b;
    }

    // 规定对象排序的算法：先按照 a 从小到大排序；如果 a 相等，则按照 b 从小到大排序
    bool operator<(const Foo &bar) {
        if (this->a < bar.a) {
            return true;
        } else if (this->a == bar.a) {
            return this->b < bar.b;
        }
        return false;
    }

    // 规定对象排序的算法：先按照 a 从大到小排序；如果 a 相等，则按照 b 从大到小排序
    bool static decrease(const Foo &foo1, const Foo &foo2) {
        if (foo1.a > foo2.a) {
            return true;
        } else if (foo1.a == foo2.a) {
            return foo1.b > foo2.b;
        }
        return false;
    }

    friend inline ostream & operator<<(ostream &out, Foo &foo) {
        out << foo.a << " " << foo.b << endl;
        return out;
    }
};

int main() {
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    minstd_rand0 generator(seed);    // minstd_rand0 is a standard linear_congruential_engine

    vector<Foo> myVec(10, Foo());

    for (Foo &foo : myVec) {         // 随机赋值
        foo.a = generator() % 5;
        foo.b = generator() % 5;
        cout << foo;
    }

    sort(myVec.begin(), myVec.end()); // 排序一：默认从小到大，调用 operator <
    cout << endl << "after sorting using operator <" << endl;
    for (Foo &foo : myVec) {
        cout << foo;
    }

    sort(myVec.begin(), myVec.end(), Foo::decrease);  // 排序二：按照 Foo::decrease 的规则从大到小排序
    cout << endl << "after sorting using Foo::decrease()" << endl;
    for (Foo &foo : myVec) {
        cout << foo;
    }

    sort(myVec.begin(), myVec.end(), [](const Foo &foo1, const Foo &foo2) {
        // 排序三：使用 lambda 的方式进行排序，排序的方法和 Foo::decrease 一样
        if (foo1.a > foo2.a) {
            return true;
        } else if (foo1.a == foo2.a) {
            return foo1.b > foo2.b;
        }
        return false;
    }   );
    cout << endl << "after sorting using lambda" << endl;
    for (Foo &foo : myVec) {
        cout << foo;
    }

    system("pause");
    return 0;
}
