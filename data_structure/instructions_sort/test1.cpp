#include <iostream>
using namespace std;

void dfs(int pos, int cnt, int n, int k, int a[],bool visited[]) {
    //已标记了k个数，输出结果
    if (cnt == k) {
        for (int i = 0; i < n; i++)
            if (visited[i]) cout << a[i] << ' ';
        cout << endl;
        return;
    }

    //处理到最后一个数，直接返回
    if (pos == n) return;

    //如果a[pos]没有被选中
    if (!visited[pos]) {
        //选中a[pos]
        visited[pos] = true;
        //处理在子串a[pos+1, n-1]中取出k-1个数的子问题
        dfs(pos + 1, cnt + 1, n, k, a,visited);
        //回溯
        visited[pos] = false;   
    }
    //处理在子串a[pos+1, n-1]中取出k个数的问题
    dfs(pos + 1, cnt, n, k, a, visited);
}
int main() {
    int i, n, k;
    while (cin >> n >> k, n || k) 
    {
        int *a = new int[n];
        bool *visited = new bool[n];
        for (i = 0; i < n; i++)
        {
            a[i] = i + 1;
            visited[i] = false;
        }
        dfs(0, 0, n, k, a, visited);
        delete[] a;
        delete[] visited;
    }
    //getchar();
    return 0;
}