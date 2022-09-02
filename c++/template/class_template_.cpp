#include <iostream>

template <class T> class matrix {
    public:
        matrix();
        ~matrix();
        resize(int height, int weight);
    private:
        T *m_pT;        
        int m_height;
        int m_weight;
};

template <class  T>  matrix<T>::matrix(){
   m_height = 100;
   m_weight = 100;
   m_pT = new T[m_weight][m_height];
}
template <class T>  matrix<T>::~matrix() {
   delete [][]m_pT ;
}
        
template <class T> void matrix<T>::resize(int height, int weight) {
    delete [][]m_pT;
    m_pT = new T[height][weitht];
    m_hegiht = height;
    m_weight = weight;
}

   m_weight * height + weight;




int main() {
    matrix<int> intMatrix;
    intStack.push(1);
    intStack.push(2);
    intStack.push(3);
    
    while (!intStack.isEmpty()) {
        std::cout << " " << intStack.pop();
    }
    std::cout << std::endl;
    return 0;
}
