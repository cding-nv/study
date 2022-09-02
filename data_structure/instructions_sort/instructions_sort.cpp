/* 题目

有N条指令， I1，I2，….IN，以及指令间的关系组合：
（Ii， Ij， Ck）： 表示Ii 与Ij有依赖，Ij需要等待Ii发射后Ck的时间（cycles）才能发射。
每个cycle只能发射一条指令，需要排出时间最短的指令发射序列

比如有3条指令 I1, I2, I3
关系集合：（I1， I3， 1） （I2， I3， 2）
假设起始时间是0， 那么可以获得的时间最短序列为：
I2  I1  I3     （I2：0  I1 ：1  I3：2， 总时间为2）
想像一下，如果排列为 I1  I2  I3， 那么总时间至少为3
*/

// How to run:
// $ g++ ./instructions_sort.cpp -o instructions_sort
// $ ./instructions_sort

// 解题思路
//  1. next_permutation 做全排列
//  2. 去除不合法的排列, 比如 I3 依赖 I1, 那么 I3 必须在 I1 的后面
//  3. 对于合法的排列, 计算所需要的时间,这里只计算 需要插入 Stall  cycle 的个数
//     a. Stall cycle 必然是由关系集合里的依赖关系引起的
//     b. 比如指令序列 2 1 3
//        1, 3 有依赖, 距离是0, 需要插入一个cycle, 即数组变为  2, 1, 0, 3
//        2, 3 有依赖, 距离是2, 不再需要插入stall cycle
//       Notes:   Debug中发现 std::find() 对于不同的数组元素竟然然返回了相同的指针地址, 还不得其解, 只好又单独实现了 find_i() 返回数组下标
//     c.  插入的 stall cycle 越少, 总的时间就越少
//  4. 优化
//       a. 两组 for/for 循环应该可以融合在一起
//       b. 不用全排列, 给有依赖关系的指令按照等待时间从小到大排序, 先排无依赖的指令, 再从小到大往两头排有依赖的指令,
//       c. 乱序的经典算法有记分牌技术和 Tomasulo 算法

//  Chris Ding 2020.3.26

#include <iostream>
#include <algorithm>
#include <stdio.h>

using namespace std;

#define N 3

//
int find_i(int ar[], int n, int element) {
    int i = 0;
    int index=-1;
    for (i = 0; i <n; i++) {
        if (element ==ar[i]) {
            index=i;
        }
    }
    return index;
}

int main () {
    //int n = N;
    int instructions[N] = {1, 2, 3};
    for (int i = 0; i < N; i++) {
        instructions[i] = i + 1;
    }

    // Relationship set
    int relate_set[N + 1][N + 1] = {0};
    relate_set[1][3] = 1;
    relate_set[2][3] = 2;

    int cost = 0xffff;
    int min_cost_order[N] = {0};

    do {
        cout << instructions[0] << ' ' << instructions[1] << ' ' << instructions[2] <<'\n';

        bool flag_valid = true;

        for (int i = 1; i < N + 1; i++) {
            for (int j = 1; j < N + 1; j++) {
                //cout << "    i j " << i << " " << j << endl;
                if (relate_set[i][j] > 0) {
                    cout << "    relate_set[" << i << "][" << j << "] = " << relate_set[i][j] << endl;
                    int* position_i = find(instructions, instructions + N - 1, i);
                    int* position_j = find(instructions, instructions + N - 1, j);
                    if (position_i > position_j) {
                        flag_valid = false;
                        cout << "    " << instructions[0] << " " << instructions[1] << " " << instructions[2] << " is false" << endl;
                        break;
                    }
                }
            }
            if (flag_valid == false) break;
        }
        if (flag_valid) {
            int temp_cost = 0;
            // Insert stall cycle.
            int temp_instr[128] = {0};
            for (int i = 0; i < N; i++) {
                temp_instr[i] = instructions[i];
            }
            // Only count waiting cycles and save in temp_cost
            for (int i = 1; i < N+1; i++)
                for (int j = 0; j < N+1; j++) {
                    if (relate_set[i][j] > 0) {
                        printf("   : [%d][%d]\n", i, j);
                        int index_i = find_i(temp_instr, N+temp_cost, i);
                        int index_j = find_i(temp_instr, N+temp_cost, j);
                        printf("     index_i, index_j %d  %d  %d \n", index_i, index_j, index_j - index_i);
                        int distance = index_j - index_i - 1;
                        printf("    %d %d distance = %d\n", i, j, distance);

                        if (relate_set[i][j] - distance > 0) {
                            temp_cost += relate_set[i][j] - distance;

                            // Rear shift from j and insert stall cycle before j.
                            for (int k = N + temp_cost; k > j - 1; k--) {
                                temp_instr[k] = temp_instr[k-1];
                            }
                            temp_instr[j-1] = 0;
                        }


                    }
                }
            cout << "  temp_cost = " << temp_cost << endl;
            if (temp_cost < cost) {
                cost = temp_cost;
                cout << "  cost = " << cost << endl;
                for (int i = 0; i < N; i++) {
                    min_cost_order[i] = instructions[i];
                }
            }
        }
    } while ( next_permutation(instructions , instructions + N) );

    cout << "min_cost_order " << endl;
    for (int i = 0; i < N; i++) {
        cout << " " << min_cost_order[i];
    }
    cout << endl;

    return 0;
}