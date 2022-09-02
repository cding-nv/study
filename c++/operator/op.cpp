#include <iostream>

class person {
private:
    int age;

public:
    person(int a) {
        this->age = a;
    }
    
    inline bool operator == (const person &ps) const;
};

inline bool person::operator==(const person &ps) const {
    if (this->age != ps.age) {
        std::cout << "this age , ps age = " << this->age << "," << ps.age << std::endl;
        return true;   
    }
    return false;
}

int main() {
    person p1(10);
    person p2(20);
    if (p2 == p1) {
        std::cout << "the age is equal" << std::endl;
    } else {
        std::cout << "the age is not equal" << std::endl;
    }
    return 0;
}