// Timer 0: 13(8) & 4(8)
// Timer 1: 11(16) & 12(16)
// Timer 2: 10(8) & 9(8)
// Timer 3: 5,2,3 (all 16)
// Timer 4: 6,7,8 (all 16)
// Timer 5: 44,45,46 (all 16)

#define WHITE 11  // timer 1
#define BLUE 5    // timer 3
#define RED 6     // timer 4
#define GREEN 44  // timer 5

uint16_t led_hertz[4] = {2,2,2,2};

uint8_t led_pins[4] = {11,5,6,44};

uint8_t is_start[4] = {0,0,0,0};

uint8_t randNumber[4] = {0,0,0,0};

uint8_t attempt = 0;
uint8_t max_tries = 5;

void setup_timer0(uint16_t hertz);
void setup_timer1(uint16_t hertz);
void setup_timer2(uint16_t hertz);
void setup_timer3(uint16_t hertz);
void setup_timer4(uint16_t hertz);
void setup_timer5(uint16_t hertz);
void setup_timers(uint16_t hertz0, uint16_t hertz1, uint16_t hertz2, 
  uint16_t hertz3, uint16_t hertz4, uint16_t hertz5);
void reset_all_timers();

void setup(){
  // Setup pin modes
  pinMode(WHITE, OUTPUT);
  pinMode(BLUE, OUTPUT);
  pinMode(RED, OUTPUT);
  pinMode(GREEN, OUTPUT);

  //Setup terminal
  Serial.begin(9600);
  
  randomSeed(2020);
  
  cli();//stop interrupts

  setup_timers(0, led_hertz[0], 0, led_hertz[1], led_hertz[2], led_hertz[3]);

  sei();

}//end setup

ISR(TIMER1_COMPA_vect){
  if (is_start[0])
    digitalWrite(WHITE, !digitalRead(WHITE));
  else
    digitalWrite(WHITE, LOW);
}

ISR(TIMER3_COMPA_vect){
  if (is_start[1])
    digitalWrite(BLUE, !digitalRead(BLUE));
  else
    digitalWrite(BLUE, LOW);
}
  
ISR(TIMER4_COMPA_vect){
  if (is_start[2])
    digitalWrite(RED, !digitalRead(RED));
  else
    digitalWrite(RED, LOW);
}

ISR(TIMER5_COMPA_vect){
  if (is_start[3])
    digitalWrite(GREEN, !digitalRead(GREEN));
  else
    digitalWrite(GREEN, LOW);
}

void setup_timer0(uint16_t hertz) {}

void setup_timer1(uint16_t hertz) {
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1  = 0;
  TIMSK1 = 0;
  // set compare match register for 2Hz increments
  OCR1A = 16000000 / (hertz*256) - 1;
  // turn on CTC mode
  TCCR1B |= (1 << WGM12);
  // prescaler = 256 
  TCCR1B |= (1 << CS12) | (0 << CS11) | (0 << CS10);   
  // enable timer compare interrupt
  TIMSK1 |= (1 << OCIE1A);
}

void setup_timer2(uint16_t hertz) {}

void setup_timer3(uint16_t hertz) {
  TCCR3A = 0;
  TCCR3B = 0;
  TCNT3  = 0;
  // set compare match register for 2Hz increments
  OCR3A = 16000000 / (hertz*256) - 1;
  // turn on CTC mode
  TCCR3B |= (1 << WGM32);
  // prescaler = 256 
  TCCR3B |= (1 << CS32) | (0 << CS31) | (0 << CS30);   
  // enable timer compare interrupt
  TIMSK3 |= (1 << OCIE3A);
}

void setup_timer4(uint16_t hertz) {
  TCCR4A = 0;
  TCCR4B = 0;
  TCNT4  = 0;
  // set compare match register for 2Hz increments
  OCR4A = 16000000 / (hertz*256) - 1;
  // turn on CTC mode
  TCCR4B |= (1 << WGM42);
  // prescaler = 256 
  TCCR4B |= (1 << CS42) | (0 << CS41) | (0 << CS40);   
  // enable timer compare interrupt
  TIMSK4 |= (1 << OCIE4A);
}

void setup_timer5(uint16_t hertz) {
  TCCR5A = 0;
  TCCR5B = 0;
  TCNT5  = 0;
  // set compare match register for 2Hz increments
  OCR5A = 16000000 / (hertz*256) - 1;
  // turn on CTC mode
  TCCR5B |= (1 << WGM52);
  // prescaler = 256 
  TCCR5B |= (1 << CS52) | (0 << CS51) | (0 << CS50);   
  // enable timer compare interrupt
  TIMSK5 |= (1 << OCIE5A);
}

void setup_timers(uint16_t hertz0, uint16_t hertz1, uint16_t hertz2, 
  uint16_t hertz3, uint16_t hertz4, uint16_t hertz5) {
  setup_timer0(hertz0);
  setup_timer1(hertz1);
  setup_timer2(hertz2);
  setup_timer3(hertz3);
  setup_timer4(hertz4);
  setup_timer5(hertz5);
}

void reset_all_timers() {
  TCCR0A = 0; TCCR0B = 0; TCNT0  = 0; TIMSK0 = 0;
  TCCR1A = 0; TCCR1B = 0; TCNT1  = 0; TIMSK1 = 0;
  TCCR2A = 0; TCCR2B = 0; TCNT2  = 0; TIMSK2 = 0;
  TCCR3A = 0; TCCR3B = 0; TCNT3  = 0; TIMSK3 = 0;
  TCCR4A = 0; TCCR4B = 0; TCNT4  = 0; TIMSK4 = 0;
  TCCR5A = 0; TCCR5B = 0; TCNT5  = 0; TIMSK5 = 0;
}

void loop() {
  char line_buffer[5];
  unsigned char read_length;
  boolean is_win = false;

  randNumber[0] = (uint8_t)random(9) + 1;
  randNumber[1] = (uint8_t)random(10);
  randNumber[2] = (uint8_t)random(10);
  randNumber[3] = (uint8_t)random(10);

  Serial.println("\nWelcome to Codebreaker! Try to break the 4 digit code within 5 tries.");

  for (int i = 0; i < 4; i++) {
    Serial.print(randNumber[i]);
  }
  Serial.println();
  
  while(attempt < max_tries && !is_win)
  {
    read_length = Serial.readBytesUntil('\n', line_buffer, 4);
    if(read_length == 4)
    {
      attempt += 1;
      Serial.print("Your prediction is: ");
      line_buffer[4] = '\0';
      Serial.println(line_buffer);

      uint8_t ncorrect = 0;
      for (int i = 0; i < 4; i++) {
        if (uint8_t(line_buffer[i])-48 == randNumber[i]) {
          is_start[i] = 0;
          led_hertz[i] = 2;
          ncorrect++;
        } else {
          is_start[i] = 1;
          led_hertz[i] *= 2;
        }
      }
      setup_timers(0, led_hertz[0], 0, led_hertz[1], led_hertz[2], led_hertz[3]);

      if (ncorrect == 4)
        is_win = true;
    }
  }

  if (!is_win) {
    Serial.print("Game over. The code is: ");
    for (int i = 0; i < 4; i++) {
      Serial.print(randNumber[i]);
    }
    Serial.println();
  } else {
    Serial.println("You won!");
  }

//  cli();
  reset_all_timers();
//  sei();
  delay(0.5);

  for (int i = 0; i < 4; i++) {
    if (is_start[i]) {
      digitalWrite(led_pins[i], HIGH);
    } else {
      digitalWrite(led_pins[i], LOW);
    }
  }

  Serial.println("One more? (y/n) ");
  do { 
    char t = Serial.read();
  } while (Serial.available() > 0);
  
  while(!Serial.available()){}

  while(1) {
    char yn = 'n';
    yn = Serial.read();
    if (yn == 'y') {
      attempt = 0;
      memset(is_start, 0, sizeof(is_start));
      memset(led_hertz, 2, sizeof(led_hertz));
      setup_timers(0, led_hertz[0], 0, led_hertz[1], led_hertz[2], led_hertz[3]);
      break;
    } else if (yn == 'n') {
      Serial.println("Thanks for playing!");
      while(1);
    } else {
      
    }
  }
  
}
