#include <iostream>

class Base {
    public:
	    Base(int i) : baseI(i) {};
	    virtual void print(void) {std::cout << "call Base::print()";}
	    virtual void setI() { std::cout << "call setI()";}
	    virtual ~Base(){}
    private:
	    int baseI;
};

typedef void(*Fun)(void);

int main() {
    Base b(1000);
    int *vptrAddr = (int*) (&b);
    std::cout << "virtual vptrAddr : " << vptrAddr << std::endl;

    int virtual_table_addr = (int)* (int *) (&b);
    std::cout << "virtual_table_addr = " << virtual_table_addr << std::endl;

    //int first_virtual_fun_addr = 

    //Fun vfunc = (Fun ) virtual_table_addr;  
    //Fun vfunc = (Fun)(*(int *)*(int*)(&b));
    Fun vfunc = (Fun) (int * )(*(int*)(&b)+1);
    //Fun vfunc = (Fun) (* (int*)virtual_table_addr);
    vfunc();

    return 0;
}
