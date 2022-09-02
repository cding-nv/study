#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

// typename or class
template<class T> void swap(T& t1, T& t2) {
    T tmpT;
    tmpT = t1;
    t1 = t2;
    t2 = tmpT;
}

#if 0
template<> void swap(std::vector<int>& t1, std::vector<int>& t2) {
    t1.swap(t2);
}
#else
template<class V> void swap(std::vector<V>& t1, std::vector<V>& t2) {
    t1.swap(t2);
}
#endif


int main() {
    using namespace std;
    string str1 = "1", str2 = "2";

    swap(str1, str2);

    printf("str1:%s, str2:%s\n", str1.c_str(), str2.c_str());  
    
    vector<int> v1, v2;
    v1.push_back(1);
    v2.push_back(2);

    swap(v1, v2);

    for (int i = 0; i < v1.size(); i++) {
        printf("v1[%d]:%d\n", i, v1[i]);
    }
    for (int i = 0; i < v2.size(); i++) {
        printf("v2[%d]:%d\n", i, v2[i]);
    }
    return 0;
}
