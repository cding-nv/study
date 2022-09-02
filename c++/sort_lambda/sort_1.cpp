#include <iostream>
#include <algorithm>
#include<array>
#include<functional>


// https://en.cppreference.com/w/cpp/algorithm/sort

int main() {
    std::array<int, 10> vec = { 1,2,3,4,5,6,7,8,9 };
    std::sort(std::begin(vec ), std::end(vec ), [](int a, int b) {
        return a > b;
    });
    for (auto item : vec)
        std::cout << item << " ";

    return 0;
}
