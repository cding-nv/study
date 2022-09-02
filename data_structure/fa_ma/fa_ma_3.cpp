#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<memory.h>

/* a数组用于存储从n个整型数据中取k个数字的数组下标值*/
int a[100] = {0};

/* data数组用于存储实际的数据,也就是所有砝码的重量 * */
int data[5] = {10, 20, 50, 100, 500};

/*sum数组用于保存再data中取k个数的和,注意没有唯一化处理,也就是说可能里面存在重复唯一化处理使用函数unique; */
int sum[100] = {0};

/*index_sum用于记录sum中最后一个数据的索引值*/
int index_sum = 0;

/*这是一个递归实现,用于获取从[start,length-num]的某一位数,这个位数对应了data数组的下标,num是从
 * data中取几位数的,fujia是一个附加参数,用于记录当前获取了几位数,从而方便操作数组a */
void GetNumberNew(int start, int length, int num, int fujia) {
    for(int i = start; i <= length - num; i++) {
        if (num > 0) {
            a[num - 1] = i;
            /* 从[i+1,length]中获取num-1数 */
            GetNumberNew(i + 1, length, num - 1, fujia);
        } else {
            for(int x = 0; x < fujia; x++) {
                sum[index_sum] += data[a[x]];
            }
            index_sum++;
            return;
        }
    }
}

/* 统计长度为length的sum数组中不重复元素的个数 */
int unique(int sum[], int length) {
    int temp = index_sum;
    // printf("temp:%d ",temp);
    for(int i = 0 ; i < length - 1; i++) {
        for(int j = i + 1; j < length; j++) {
            if(sum[i] == sum[j]) {
                /*若有相同的数字则减1,并退出此次循环*/
                temp--;
                break;
            }
        }
    }
    return temp;
}

int main() {
    //data数组长度
    int length = 5;
    for(int y = 1; y <= length; y++) {
        /*从[0,num]中获取y个数*/
        GetNumberNew(0, length, y, y);
    }
    printf("%d\n", unique(sum, index_sum));
    return 0;
}
