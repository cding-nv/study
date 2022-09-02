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
    int b[4][4] = {0};
    
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    int m = 0;
    int n = 0;
    for (i = 0, m = 0; i < 4, m <4; i++, m++) {
        for (j = 0, n = 4 - 1; j < 4, n >= 0; j++, n--) {
            b[i][j] = a[n][m];        
        }
        
    }
    
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            printf("%d ", b[i][j]);
        }
        printf("\n");
    }
    
    return 0;
}