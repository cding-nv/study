#include <stdio.h>
#include <malloc.h>
int main() {
    int i1, i2, i3, i5, i10;
    int j1, j2, j3, j5, j10;
    //int a[1024];
    //int index = 0;
    printf ("input i1,i2,i3,i5,i10: ");
    scanf("%d%d%d%d%d", &i1, &i2, &i3, &i5, &i10);
    int m, n;
    m = i1 * 10 + 20 * i2 + 50 * i3 + 100 * i5 + 500 * i10 + 1;
    int *j;
    j = (int *) calloc(m, sizeof(int));
    for (j1 = 0; j1 <= i1; j1++)
        for (j2 = 0; j2 <= i2; j2++)
            for (j3 = 0; j3 <= i3; j3++)
                for (j5 = 0; j5 <= i5; j5++)
                    for (j10 = 0; j10 <= i10; j10++)
//for (j20=0;j20<=i20;j20++)
                    {
                        n = j1 * 10 + 20 * j2 + 50 * j3 + 100 * j5 + 500 * j10;
                        *(j + n) = n;
                        
                    }
    int i, total = 0;
    for (i = 0; i < m; i++)
        if ( *(i + j) > 0 ) {
            total++;
            printf("%d ", *(i+j));
        }
    printf ("%d\n", total);
    return 0;
}