#include <stdio.h>
#include <math.h>
#include <float.h>
#include <errno.h>
#include <fenv.h>
#pragma STDC FENV_ACCESS ON

int main (void) {
    printf("exp(1) = %f\n", exp(1));
    printf("ln(e) = %f\n", log(2.718282));
    printf("FV of $100, continuously compounded at 3%% for 1 year = %f\n", 100 *exp(0.03));
     
    printf("INFINITY = %f, exp(-0) = %f\n", INFINITY, exp(-INFINITY) );

    return 0;
}
