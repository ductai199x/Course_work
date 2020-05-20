#define POT A0
#define PWM1 5
#define PWM2 3

// Timer 3

uint16_t pot_reading = 0;
uint16_t original = 0;

void setup() {
  Serial.begin(19200);
  pinMode(POT, INPUT);
  pinMode(PWM1, OUTPUT);
  pinMode(PWM2, OUTPUT);
  original = TCCR3A;
}

void loop() {
  // put your main code here, to run repeatedly:
  pot_reading = analogRead(POT);
  

  if (pot_reading < 500) {
//    OCR3A = pot_reading/2;
//    OCR3B = pot_reading/2;
//    TCCR3A = 0b10110000 | (TCCR3A & 0b00001111) ;
//    TCCR3B = original;
    analogWrite(PWM1, pot_reading/2);
    analogWrite(PWM2, 255);
  } else if (pot_reading > 530){
//    OCR3A = 255-(pot_reading-512)/2;
//    OCR3B = 255-(pot_reading-512)/2;
//    TCCR3A = original;
//    TCCR3B = 0b10110000 | (TCCR3B & 0b00001111) ;
    analogWrite(PWM1, (pot_reading-512)/2);
    analogWrite(PWM2, 0);
  }
  else {
    analogWrite(PWM1, 0);
    analogWrite(PWM2, 0);
  }

  Serial.print(pot_reading);
  Serial.print(" ");
  Serial.println(OCR3A);
  
}
