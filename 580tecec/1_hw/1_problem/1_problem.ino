#define RED 5
#define GREEN 3

void setup() {
  // put your setup code here, to run once:
  pinMode(RED, OUTPUT);
  pinMode(GREEN, OUTPUT);
  digitalWrite(RED, 0);
  digitalWrite(GREEN, 0);
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(RED, HIGH);
  digitalWrite(GREEN, LOW);
  delay(100);
  digitalWrite(RED, LOW);
  digitalWrite(GREEN, HIGH);
  delay(100);
}
