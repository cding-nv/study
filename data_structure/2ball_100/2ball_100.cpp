//https://www.zhihu.com/question/31855632
https://blog.csdn.net/kingjinzi_2008/article/details/7747559

#include <iostream>
#include <cstdio>

using namespace std;

#define N 30
#define K 10000

int dp[N+5][K+5], ans = 1;

int main() {
    int n=2, m=100;
    //cin >> n >> m;
    //if (n == 1) cout << m << endl;
    
    for (int i = 1; i <= K; i++) {
        dp[1][i] = i;
    }
    for (int i = 2; i <= n; i++) {
        for (int j = 1; j <= K; j++) {
            dp[i][j] = dp[i][j-1] + dp[i-1][j-1] + 1;
        }
    }
    
    while (dp[n][ans] < m) ans++;
    
    cout << ans << endl;
    return 0;
}