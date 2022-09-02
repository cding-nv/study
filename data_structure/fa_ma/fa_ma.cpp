#include<stdio.h>
#include<math.h>

int count;

void weigh(int new_[],int n) { //筛选函数
    int i;
    int index=0;
    for(i=1; i<n; i++) {
        if(new_[index]!=new_[i]) {
            new_[++index]=new_[i];//选出不同的砝码和，放入new_函数中
        }

    }
    for(i=1; i<=index; i++)

    {
        count+=1;
        printf("%d\t",new_[i]);
        if((count%6)==0)
            printf("\n");
    }
    printf("\n");
    printf("\n");
    printf("%d",count);
}
void AllSort(int a[],int n) { //排序函数
    int i,j;
    for(i=0; i<n; i++)
        for(j=i+1; j<n; j++) {
            if(a[i]>a[j]) {
                int temp;
                temp=a[i];
                a[i]=a[j];
                a[j]=temp;
            }
        }
}

void change(int c[],int n) { //改变二进制数组次c[]中的数值，每次相当于加一
    int i;
    for(i=0; i<n; i++) {
        if(c[i])
            c[i]=0;
        else {
            c[i]=1;
            break;//每当有c[i]=1，跳出循环
        }

    }
}

void SonCollection(int b[],int c[],int new_[],int n) { //实现函数
    int m = pow(2, n);//排列可能数
    int i;
    for(i=0; i<m; i++) { //对每种可能进行实现
        int j,sum=0;
        for(j=0; j<n; j++) {
            if(c[j]) {
                sum=sum+b[j];

            }//printf("%d",c[j]);
        }
        change(c,n);
        new_[i]=sum;
    }
}

int main() {
    int w[5]= {10,20,50,100,500};
    int fama[5];    //五种砝码个数数组
    int i;
    printf("Input 5 fama num: ）：\n");
    for(i = 0; i < 5; i++)
        scanf("%d", &fama[i]);//输入每种砝码的个数

    int sum = 0;   //砝码的总个数
    for(i = 0; i < 5; i++) {
        sum = sum + fama[i];
    }
    int b[sum];//将所有砝码装进b数组
    int j = 0, count = 0;
    for(i = 0; i < 5; i++) {
        for(; j < count + fama[i]; j++)
            b[j] = w[i];
        count = j;
    }
    //int jj;
    //for (jj = 0; jj < j; jj++) {
    //    printf(" %d ", b[jj]);
    //}
    //printf("\n");
    int c[sum];
    printf("sum = %d\n", sum);
    for(i=0; i < sum; i++)
        c[i] = 0;//将c数组初始化为0
    int n = pow(2, sum);//pow函数用于计算2的sum次幂，共有2^sum种可能，包括重复的和数
    int new_[n];
    SonCollection(b, c, new_, sum);
    AllSort(new_, n);
    weigh(new_, n);
    return 0;
}