//https://blog.csdn.net/xiaolewennofollow/article/details/52559306
 
#include <iostream>
#include <vector>
#include <string> 
#include <string.h>
 
 class MyString { 
 private: 
  char* _data; 
  size_t   _len; 
  void _init_data(const char *s) { 
    _data = new char[_len+1]; 
    memcpy(_data, s, _len); 
    _data[_len] = '\0'; 
  } 
 public: 
  MyString() { 
    _data = NULL; 
    _len = 0; 
  } 

  MyString(const char* p) { 
    _len = strlen (p); 
    _init_data(p); 
  } 

  MyString(const MyString& str) { 
    _len = str._len; 
    _init_data(str._data); 
    std::cout << "Copy Constructor is called! source: " << str._data << std::endl; 
  } 

  MyString& operator=(const MyString& str) { 
    if (this != &str) { 
      _len = str._len; 
      _init_data(str._data); 
    } 
    std::cout << "Copy Assignment is called! source: " << str._data << std::endl; 
    return *this; 
  } 

#if 0
  MyString& operator=(MyString&& str) { 
    std::cout << "Move Assignment is called! source: " << str._data << std::endl; 
    if (this != &str) { 
      _len = str._len; 
      _data = str._data; 
      str._len = 0; 
      str._data = NULL; 
    } 
    return *this; 
 }

#endif

  virtual ~MyString() { 
    if (_data) free(_data); 
  } 
 }; 

 int main() { 
  MyString a;
  std::cout << "line2" << std::endl;  
  a = MyString("Hello"); 
  std::cout << "line4" << std::endl;
  std::vector<MyString> vec;
  std::cout << "line6" << std::endl;  
  vec.push_back(MyString("World")); 
  std::cout << "line8" << std::endl;
  vec.emplace_back("peak");
 }
 
 //MyString(“Hello”) 和 MyString(“World”) 都是临时对象，也就是右值
 //虽然它们是临时的，但程序仍然调用了拷贝构造和拷贝赋值，造成了没有意义的资源申请和释放的操作。
 //如果能够直接使用临时对象已经申请的资源，既能节省资源，有能节省资源申请和释放的时间。这正是定义转移语义的目的
 //有了右值引用和转移语义，我们在设计和实现类时，对于需要动态申请大量资源的类，应该设计转移构造函数和转移赋值函数，以提高应用程序的效率