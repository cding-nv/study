#include <iostream>

template <class T,int maxsize = 100> class Stack {
    public:
        Stack();
        ~Stack();
        void push(T t);
        T pop();
        bool isEmpty();
    private:
        T *m_pT;        
        int m_maxSize;
        int m_size;
};

template <class T,int maxsize> Stack<T, maxsize>::Stack(){
   m_maxSize = maxsize;      
   m_size = 0;
   m_pT = new T[m_maxSize];
}
template <class T,int maxsize>  Stack<T, maxsize>::~Stack() {
   delete [] m_pT ;
}
        
template <class T,int maxsize> void Stack<T, maxsize>::push(T t) {
    m_size++;
    m_pT[m_size - 1] = t;
    
}
template <class T,int maxsize> T Stack<T, maxsize>::pop() {
    T t = m_pT[m_size - 1];
    m_size--;
    return t;
}
template <class T,int maxsize> bool Stack<T, maxsize>::isEmpty() {
    return m_size == 0;
}


int main() {
    int maxsize = 10;
    Stack<int,1024> intStack;
    for (int i = 0; i < maxsize; i++) {
        intStack.push(i);
    }
    while (!intStack.isEmpty()) {
        std::cout << intStack.pop() << " ";
    }
    std::cout << std::endl;
    return 0;
}

