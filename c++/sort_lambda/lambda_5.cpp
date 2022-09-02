#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int main() {
  vector<int> vec = {1, 2, 3, 4, 5};
  auto fun = [](int num){ cout << num << endl;};
  for_each(vec.begin(), vec.end(), fun);

  // must pass reference
  for_each(vec.begin(), vec.end(), [](int& num){ num += 1;});
  for_each(vec.begin(), vec.end(), fun);
  return 0;
}
