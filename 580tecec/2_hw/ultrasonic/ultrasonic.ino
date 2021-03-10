const int trigPin = 4;
const int echoPin = 2;

int duration;
float distance;

void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  Serial.begin(9600);
}

void loop() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH);
//  distance = (duration*.0343)/2;
  distance = (duration/2)/29.1;
  Serial.print("Sensor A  ");
  Serial.print(duration);
  Serial.print('\t');
  Serial.print(distance, 4);
  Serial.println("cm");
  delay(100);
}
