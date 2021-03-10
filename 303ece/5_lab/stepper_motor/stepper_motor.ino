

// Arduino stepper motor control code

#include <Stepper.h> // Include the header file

// change this to the number of steps on your motor
#define STEPS 32

#define buttonPin 9

// create an instance of the stepper class using the steps and pins
Stepper stepper(STEPS, 2, 4, 3, 5);

int val = 60;
int buttonState = 0;

void setup() {
  Serial.begin(9600);
  stepper.setSpeed(400);
  pinMode(buttonPin, INPUT);
}

void loop() {
  buttonState = digitalRead(buttonPin);
  if(buttonState) {
    stepper.step(val);
  } else {
    stepper.step(-val);
  }

}
