#include <iostream>

int main(void) {
  // Function point
  [](){ std::cout << "1. Hello World!" << std::endl; };
  // Call function
  [](){ std::cout << "2. Hello World!" << std::endl; }();

  auto fun1 = [](){ std::cout << "3. Hello World!" << std::endl; };
  fun1();

  int num = 100;    
  auto fun2 = [](int num){ num = 5; std::cout << "4. " << num << std::endl; };
  fun2(num);
  // num = 100
  std::cout << "5. " << num << std::endl;

  return 0;
}
