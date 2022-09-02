#include <stdio.h>
#include <algorithm>
#include <iostream>

#define N 3

int main() {
    int temp_instr[5] = {2, 1, 4, 3, 0};
    int position_2 = std::find(temp_instr, temp_instr + N - 1, 2);
    int position_3 = std::find(temp_instr, temp_instr + N - 1, 3);
    int position_1 = std::find(temp_instr, temp_instr + N - 1, 1);
    int position_4 = std::find(temp_instr, temp_instr + N - 1, 4);
    printf(" position_2, position_3 %d  %d  %d \n", position_2, position_3, position_3 - position_2);
    printf(" position_1, position_3 %d  %d  %d \n", position_1, position_3, position_3 - position_1);
    printf(" position_4, position_3 %d  %d  %d \n", position_4, position_3, position_3 - position_4);
    return 0;
}
