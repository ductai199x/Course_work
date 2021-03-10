#include<inttypes.h>
#include<stdio.h>

#define FIXED_BITS        32
#define FIXED_WBITS       24
#define FIXED_FBITS       8
#define FIXED_TO_INT(a)   ((a) >> FIXED_FBITS)
#define FIXED_FROM_INT(a) (int32_t)((a) << FIXED_FBITS)
#define FIXED_MAKE(a)     (int32_t)((a*(1 << FIXED_FBITS)))
 
int32_t a, b, c;
float x, y, z;
 
static int32_t FIXED_Mul(int32_t a, int32_t b) {
  return(((int32_t)a*(int32_t)b) >> FIXED_FBITS);
} 
int main(void) {
  //floating point
  x = 8.0;  //11 clock cycles
  y = 2.5;  //12 clock cycles
  z = x*y;  //1651 clock cycles (104.69us)
  //fixed point
  a = FIXED_MAKE(8.0);  //11 clock cycles
  b = FIXED_MAKE(2.5111);  //12 clock cycles
  c = FIXED_Mul(a, b);  //175 clock cycles (10.94us)

  printf("%d\n", b);
  return 0;
}