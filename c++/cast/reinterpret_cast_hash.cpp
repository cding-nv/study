// expre_reinterpret_cast_Operator.cpp
// compile with: /EHsc
#include <iostream>

// Returns a hash code based on an address
unsigned short Hash( void *p ) {
	unsigned long val = reinterpret_cast<unsigned long>( p );
	return ( unsigned short )( val ^ (val >> 16));
}

using namespace std;

int main() {
    cout << "Hello world. " << std::endl;
	int a[4];
	for ( int i = 0; i < 4; i++ )
		cout << Hash( a + i ) << std::endl;
    return 0;
}