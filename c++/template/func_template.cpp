#include <iostream>

template<typename T> void swap(T& t1, T& t2);

template<typename T> void swap(T& t1, T& t2) {
    T tmpT;
    tmpT = t1;
    t1 = t2;
    t2 = tmpT;
}

int main() {
    int num1 = 1, num2 = 2;
    swap<int>(num1, num2);
    std::cout << "num1 num2: " <<  num1 << " , " << num2 << std::endl;  
    return 0;
}
