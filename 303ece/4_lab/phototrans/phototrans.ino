int photo_pin = A0; 
int val1 = 0;
unsigned int val = 0; 
unsigned int counter = 0;

void setup() {
  Serial.begin(9600);
  pinMode(5, OUTPUT);
}

unsigned int avg = 0;

void loop() {
  if (Serial.available() > 0) {
    val = Serial.parseInt();
    avg = 0;
    for (int i = 0; i < 5; i++) {
      analogWrite(5, counter);
      delay(100);
      avg += analogRead(photo_pin);
    }
    
    Serial.println(avg);
    counter += 1;
  }

}
