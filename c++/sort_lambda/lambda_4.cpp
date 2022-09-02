#include <iostream>

using namespace std;

int main(void)
{
  string str;
  auto fun = [](string str){cout << str << endl; }; 
  cin >> str;
  fun(str);
  return 0;
}
