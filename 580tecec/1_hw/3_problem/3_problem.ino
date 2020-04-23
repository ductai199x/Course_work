#define VIN A0
#define RED 5
#define GREEN 3

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(VIN, INPUT);
  pinMode(RED, OUTPUT);
  pinMode(GREEN, OUTPUT);
  digitalWrite(RED, 0);
  digitalWrite(GREEN, 0);
}

void loop() {
  int reading = analogRead(VIN);
  float voltage = reading * 5.0f/1024;
  Serial.print(reading);
  Serial.print(" ~ ");
  Serial.print(voltage, 4);
  Serial.print("\n");

  if (voltage < 2) {
    digitalWrite(GREEN, LOW);
    digitalWrite(RED, HIGH);
    delay(500);
    digitalWrite(RED, LOW);
    delay(500);
  } else if (voltage <= 3) {
    digitalWrite(GREEN, LOW);
    digitalWrite(RED, LOW);
    delay(1000);
  } else {
    digitalWrite(RED, LOW);
    digitalWrite(GREEN, HIGH);
    delay(500);
    digitalWrite(GREEN, LOW);
    delay(500);
  }
}
