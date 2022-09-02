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
