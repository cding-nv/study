#include <iostream>

using namespace std;

template <class T>
 class compare
 {
  public:
  bool equal(T t1, T t2)
  {
       return t1 == t2;
  }
};

 int main()
 {
  
  char str1[] = "Hello";
  char str2[] = "Hello";
  
  compare<int> c1;
  compare<char *> c2;   
  cout << c1.equal(1, 1) << endl;        // compare 2 int
  cout << c2.equal(str1, str2) << endl;   // compare 2 char
  return 0;
 }
