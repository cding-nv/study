#include <stdio.h>

int main() {
    int i = 0;
    int j = 0;
    int tmp = 0;
    
    int a[4][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };
    
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }
    
    for (i = 0; i < 4; i++) {
        for (j = i; j < 4; j++) {
            tmp = a[j][i];
            a[j][i] = a[i][j];
            a[i][j] = tmp;
        }
    }
    
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }
    
    return 0;
}