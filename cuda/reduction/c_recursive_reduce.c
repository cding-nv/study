#include <stdio.h>

// Recursive Implementation of Interleaved Pair Approach
 int cpuRecursiveReduce(int *data, int const size)
 {
     // stop condition
     if (size == 1) return data[0];
     // renew the stride
     int const stride = size / 2;
     // in-place reduction
     for (int i = 0; i < stride; i++)
     {
         data[i] += data[i + stride];
     }
     // call recursively
     return cpuRecursiveReduce(data, stride);
}

int main() {
    int a[] = {2, 4, 6, 8};
    int n = sizeof(a) / sizeof(a[0]);
    int sum = 0;

    printf("size = %d\n", n);
    
    sum = cpuRecursiveReduce(a, n);
    printf("sum = %d\n", sum);

    
    return 0;

}
