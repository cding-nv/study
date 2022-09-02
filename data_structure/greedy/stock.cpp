#include <iostream>

using namespace std;

int maxProfit(int price[], int len) {
    int profit = 0;
    int tmp = 0;
    cout << "len " << len << endl;
    for (int i = 1; i < len; i++) {
        tmp = price[i] - price[i-1];
        if (tmp > 0) {
            profit += tmp;
        }
    }
    return profit;
}

int main() {
    //int a[6] = {7, 1, 5, 3, 6, 4};
    //int a[] = {1, 2, 3, 4, 5};
    int a[] = {7, 6, 4, 3, 1};
    int len = sizeof(a) / sizeof(a[0]);
    int profit = maxProfit(a, len);
    cout << "profit " << profit << endl;
    return 0;
}

