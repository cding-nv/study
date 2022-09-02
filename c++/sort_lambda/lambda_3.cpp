#include <iostream>

int main(void) {
  int num = 100;
  // read only 
  auto fun1 = [=](){ std::cout << num << std::endl; };
  fun1();

  // read and write
  auto fun2 = [&num](){ num = 200; std::cout << num << std::endl; };
  fun2();
  // num = 200
  std::cout << num << std::endl;
  return 0;
}
