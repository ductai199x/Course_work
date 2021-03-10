#define VIN A0

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(VIN, INPUT);
}

void loop() {
  int reading = analogRead(VIN);
  float voltage = reading * 5.0f/1024;
  Serial.print(reading);
  Serial.print(" ~ ");
  Serial.print(voltage, 4);
  Serial.print("\n");
  delay(500);
}
