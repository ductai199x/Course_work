
volatile float x,y,z;
volatile int8_t a,b,c;

unsigned long t1, t2;

void setup(){
  Serial.begin(9600);

  uint32_t n;
  uint32_t max_it = 10000;
  
  Serial.print("Floating point 6.5625*4.25: ");
  n = 0;
  t1 = micros();
  while (n < max_it) {
    x = 6.5625;
    y = 4.25;
    z = x*y;
    n++;
  }

  t2 = micros();
  Serial.println((t2-t1)/max_it);

  Serial.print("Fixed point 6.5625*4.25 (W=8): ");
  n = 0;
  t1 = micros();
  while (n < max_it) {
    a = (int8_t)(6.5625*(1 << 3)); // 3Q4
    b = (int8_t)(4.25*(1 << 5));   // 5Q2
    c = (int8_t)(a*b);  // 8Q6 cast into int8
    n++;
  }
  t2 = micros();
  Serial.println((t2-t1)/max_it);

  Serial.print("Floating point 0.5625*30.25: ");
  n = 0;
  t1 = micros();
  while (n < max_it) {
    x = 0.5625;
    y = 30.25;
    z = x*y;
    n++;
  }

  t2 = micros();
  Serial.println((t2-t1)/max_it);

  Serial.print("Fixed point 0.5625*30.25 (W=8): ");
  n = 0;
  t1 = micros();
  while (n < max_it) {
    a = (int8_t)(0.5625*(1 << 3)); // 3Q4
    b = (int8_t)(30.25*(1 << 5));   // 5Q2
    c = (int8_t)(a*b);  // 8Q6 cast into int8
    n++;
  }
  t2 = micros();
  Serial.println((t2-t1)/max_it);

  Serial.print("Floating point -0.5625*30.25: ");
  n = 0;
  t1 = micros();
  while (n < max_it) {
    x = -0.5625;
    y = 30.25;
    z = x*y;
    n++;
  }

  t2 = micros();
  Serial.println((t2-t1)/max_it);

  Serial.print("Fixed point -0.5625*30.25 (W=8): ");
  n = 0;
  t1 = micros();
  while (n < max_it) {
    a = (int8_t)(-0.5625*(1 << 3)); // 3Q4
    b = (int8_t)(30.25*(1 << 5));   // 5Q2
    c = (int8_t)(a*b);  // 8Q6 cast into int8
    n++;
  }
  t2 = micros();
  Serial.println((t2-t1)/max_it);

  Serial.print("Fixed point 6.5625*4.25 (W=16): ");
  n = 0;
  t1 = micros();
  while (n < max_it) {
    a = (int16_t)(6.5625*(1 << 3));
    b = (int16_t)(4.25*(1 << 5));
    c = (int16_t)(a*b);
    n++;
  }
  t2 = micros();
  Serial.println((t2-t1)/max_it);
  
}

void loop() {
  // put your main code here, to run repeatedly:

}
