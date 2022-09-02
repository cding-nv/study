#include <iostream>
using namespace std;

void Perm(int start, int end, int a[]) {
    //得到全排列的一种情况，输出结果
    if (start == end) {
        for (int i = 0; i < end; i++)
            cout << a[i] << ' ';
        cout << endl;
        return;
    }
    for (int i = start; i < end; i++) {
        swap(a[start], a[i]);      //交换
        Perm(start + 1, end, a);   //分解为子问题a[start+1,...,end-1]的全排列
        swap(a[i], a[start]);      //回溯
    }
}
int main() {
    // n instructions
    int n = 3;
    
    // Relationship set
    int relate_set[10][10] = {0};
    relate_set[1][3] = 1;
    relate_set[2][3] = 2;
    
    int a[10];
    int i = 0;
    
    while (n) {
        for (i = 0; i < n; i++)
        {
            a[i] = i + 1;
        }
        Perm(0, n, a);
    }
    return 0;
}