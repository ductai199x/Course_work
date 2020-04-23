volatile unsigned long LastPulseTime;
volatile unsigned long startTime;
int duration;

#define trigPin 4
#define echoPin 2


void setup() {
  Serial.begin (9600);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  attachInterrupt(0, EchoPin_ISR, CHANGE);  // Pin 2 interrupt on any change
}
void loop(){
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  Serial.print("Sensor A  ");
  Serial.print(LastPulseTime);
  Serial.print('\t');
  Serial.print((LastPulseTime/2) / 29.1,4);
  Serial.println("cm");

  delay(100);
}
void EchoPin_ISR() {
    if (digitalRead(echoPin)) // Gone HIGH
        startTime = micros();
    else  // Gone LOW
        LastPulseTime = micros() - startTime;
}
