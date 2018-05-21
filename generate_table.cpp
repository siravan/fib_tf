#include <cstdio>
#include <cmath>

#define __device__
#define __host__

struct float3 {
    float x, y, z;
};

#include "ionic.h"
#include "courtemanche.h"

int main() {
    float V = -50.0;
    float inter[30];

    calc_inter(V, inter);

    for(int i=0; i<30; i++) {
        printf("%f\n", inter[i]);
    }
}
