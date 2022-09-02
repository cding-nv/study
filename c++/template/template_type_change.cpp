template <class T> class Stack {
    public:
        Stack();
        ~Stack();
        void push(T t);
        T pop();
        bool isEmpty();
        template<class T2>  operator Stack<T2>();
    private:
        T *m_pT;        
        int m_maxSize;
        int m_size;
};
