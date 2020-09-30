#include <math.h>
#include "sigmoid.h"

float sigmoid(float x)
{
     float exp_value;
     float return_value;

     /*** Exponential calculation ***/
     exp_value = exp((double) -x);

     /*** Final sigmoid value ***/
     return_value = 1 / (1 + exp_value);

     return return_value;
}


float sigmoidDeriv(float x) {
    return x*(1-x);
}

