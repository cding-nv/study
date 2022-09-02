#include <iostream>
using namespace std;

int main() {
    int a[128][4];
    int n = 0;
    int over = 0;
    int total_sheet = 0;
    
    cin >> n;
    cin >> over;
    
    if (n % 4 == 0)    total_sheet = n / 4;
    else total_sheet = n / 4 + 1;
    
    cout << "Printing order for " << n << " pages:" << endl;
    
    if (n % 4 == 0) {
        for (int i = 1; i <= total_sheet; i++) {
            a[i][0] = n - (i - 1) * 2;
            a[i][1] = i * 2 - 1;
            a[i][2] = i * 2;
            a[i][3] = a[i][0] - 1;
            
            cout << "Sheet " << i << ", front: " << a[i][0] << ", " << a[i][1] << endl;
            cout << "Sheet " << i << ", back : " << a[i][2] << ", " << a[i][3] << endl;
        }
    }
    
    if (n % 4 != 0) {
        int nn  = total_sheet * 4;
        for (int i = 1; i <= total_sheet; i++) {
            a[i][0] = nn - (i - 1) * 2;   
            a[i][1] = i * 2 - 1;          
            a[i][2] = i * 2;              
            a[i][3] = a[i][0] - 1;

            for (int j = 0; j<4; j++) {
                if (a[i][j] > n) a[i][j] = 0;    
            }
            
            cout << "Sheet " << i << ", front: ";
            if (a[i][0] <= 0) cout << "NA";
            else cout << a[i][0];
            cout << ", " << a[i][1] << endl;
            
            cout << "Sheet " << i << ", back : " << a[i][2] << ", ";
            if (a[i][3]<= 0) cout << "NA";
            else cout << a[i][3];
            cout << endl;
        }
    }
    return 0;
}