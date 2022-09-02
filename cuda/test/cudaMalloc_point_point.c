#include <stdio.h>
#include<stdlib.h>

void P1Malloc(int* p) {
    p=(int*)malloc(10);
    printf("P1Malloc,current malloc address:%p\n",p);
}

void P2Malloc(void** p) {
    *p=malloc(10);
    printf("P2Malloc,current malloc address:%p\n",*p);
}

int main() {
    int Num=10;
    int* a=&Num;
    printf("initial pointer a:%p\n",a);
    P1Malloc(a);
    printf("*a = %d\n", *a);
    printf("after using *,ponter a:%p\n",a);
    P2Malloc((void**)&a);
    printf("after using **,ponter a:%p\n",a);
    printf("*a = %d\n", *a);
    return 0;
}