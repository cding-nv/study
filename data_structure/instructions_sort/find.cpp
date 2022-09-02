
#include <iostream>
#include <vector>
#include <algorithm>//注意要包含该头文件
using namespace std;
int main()
{
    int nums[] = { 3, 1, 4, 1, 5, 9 };
    int *result = find( nums , nums + 5,3 );
    int *result2 = find(nums, nums+5, 1);
    if( result == nums + 5 ) 
        cout<< "Did not find any number matching " << endl;
    else {
         cout<< "Found a matching number: " << *result << endl;
         cout << "result - result2 " << result2 - result << endl;
    }
    return 0;
}