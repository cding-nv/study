#include <stdio.h>

#define  EAST  1
#define  SOUTH 2
#define  WEST  3
#define  NORTH 4

int main() {
    int i = 0;
    int j = 0;
    int n = 0;
    
    int a[5][4] = {
        {1,  2,  3,  4},
        {12, 13, 14, 5},
        {11, 16, 15, 6},
        {10, 9,  8,  7},
        {1,  1,  1,  1}
    };

    int status = EAST;
    int row_max = sizeof(a) / sizeof(a[0]);
    int row_min = -1;
    int column_max = sizeof(a[0]) / sizeof(a[0][0]);
    int column_min = -1;
    int total = row_max * column_max;
    printf("row/column : %d / %d\n", row_max, column_max);

    for (n = 0; n < total;  n++) {
        printf(" %d ", a[i][j]);
        if (status == EAST) j++;
        if (status == SOUTH) i++;
        if (status == WEST) j--;
        if (status == NORTH) i--;

        if (j == column_max) {
            status = SOUTH;
            row_min++;
            j--;
            i++;
        }
        if (i == row_max) {
            status = WEST;
            column_max--;
            i--;
            j--;
        }
        if (j == column_min) {
            status = NORTH;
            row_max--;
            j++;
            i--;
        }
        if (i == row_min) {
            status = EAST;
            column_min++;
            j++;
            i++;
        }
        
    }

    return 0;
}