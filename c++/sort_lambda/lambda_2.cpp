#include <iostream>

int main(void) {
  int num1 = [](int a, int b){ return a + b; }(1, 2);
  int num2 = [](double a, double b)->int{ return a + b; }(1.2, 2.1);
  double num3 = [](double a, double b)->decltype(a + b){ return a + b; }(1.2, 2.1);
  std::cout << num1 << std::endl;
  std::cout << num2 << std::endl;
  std::cout << num3 << std::endl;
  return 0;
}
