#include<stdio.h>
#define MAX_SIZE 10
 
void merging(int *l, int l_size, int *r, int r_size)
{
    int i, j, k, temp[MAX_SIZE];
    i=j=k=0;
    while(i<l_size && j<l_size)
    {   if(l[i]<r[j])
        {
            temp[k++]=l[i++];
        }
        else
        {
            temp[k++]=r[j++];
        }
    }
    while(i<l_size)
    {
        temp[k++]=l[i++];
    }
    while(j<r_size)
    {
        temp[k++]=r[j++];
    }
 
    for(i=0;i<l_size+r_size;i++)
    {
        l[i]=temp[i];
    }
}
 
void mergesort(int a[], int n)
{
    int *l=a;
    int *r=a+n/2;
    int l_size=n/2;
    int r_size=n-l_size;
 
    if(n>1)
    {
        mergesort(l, l_size);
        mergesort(r, r_size);
        merging(l,l_size,r,r_size);
    }
}
 
int main(void)
{
    int i;
    int a[10]={-1,5,2,6,0,3,9,1,7,4};
    printf("排序前：");
    for(i=1;i<10;i++)
    {
        printf("%d",a[i]);
    }
    mergesort(a, 9);
    //printf("\n\n共交换数据%d次\n\n", c);
    printf("排序后：");
    for(i=1;i<10;i++)
    {
        printf("%d",a[i]);
    }
    printf("\n\n\n");
    return 0;

}
