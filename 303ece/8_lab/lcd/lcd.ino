/*
  LiquidCrystal Library - Hello World

 Demonstrates the use a 16x2 LCD display.  The LiquidCrystal
 library works with all LCD displays that are compatible with the
 Hitachi HD44780 driver. There are many of them out there, and you
 can usually tell them by the 16-pin interface.

 This sketch prints "Hello World!" to the LCD
 and shows the time.

  The circuit:
 * LCD RS pin to digital pin 12
 * LCD Enable pin to digital pin 11
 * LCD D4 pin to digital pin 5
 * LCD D5 pin to digital pin 4
 * LCD D6 pin to digital pin 3
 * LCD D7 pin to digital pin 2
 * LCD R/W pin to ground
 * LCD VSS pin to ground
 * LCD VCC pin to 5V
 * 10K resistor:
 * ends to +5V and ground
 * wiper to LCD VO pin (pin 3)

 Library originally added 18 Apr 2008
 by David A. Mellis
 library modified 5 Jul 2009
 by Limor Fried (http://www.ladyada.net)
 example added 9 Jul 2009
 by Tom Igoe
 modified 22 Nov 2010
 by Tom Igoe
 modified 7 Nov 2016
 by Arturo Guadalupi

 This example code is in the public domain.

 http://www.arduino.cc/en/Tutorial/LiquidCrystalHelloWorld

*/

// include the library code:
#include <LiquidCrystal.h>
#include "dht.h"

// initialize the library by associating any needed LCD interface pin
// with the arduino pin number it is connected to
//const int rs = 12, en = 11, d4 = 5, d5 = 4, d6 = 3, d7 = 2;
LiquidCrystal lcd(43, 45, 41, 39, 37, 35);

#define DHTPIN 9 // Analog Pin sensor is connected to
#define DHTTYPE DHT11

#define WATER_SENSOR_PIN A5
 
DHT dht(DHTPIN, DHTTYPE, 10);

void setup() {
  // set up the LCD's number of columns and rows:
  lcd.begin(16, 2);
  // Print a message to the LCD.
  lcd.print("hello, world!");

  pinMode(WATER_SENSOR_PIN, INPUT);

  Serial.begin(9600);
  Serial.println("DHT11 Humidity & temperature Sensor\n\n");
  delay(3000);//Wait before accessing Sensor
}

float temp = 0.0f;
float humid = 0.0f;
float water_level = 0.0f;

void loop() {
  // set the cursor to column 0, line 1
  // (note: line 1 is the second row, since counting begins with 0):
  
  // print the number of seconds since reset:
  int chk = dht.read(DHTPIN);

  humid = dht.readHumidity();
  temp = dht.readTemperature(true);
  water_level = (float)analogRead(WATER_SENSOR_PIN)/1024*100;

  lcd.clear();
  lcd.setCursor(0, 0);
  
  Serial.print(temp, 2);
  Serial.print(" ");
  Serial.print(humid, 2);
  Serial.print(" ");
  Serial.println(water_level, 2);
  
  lcd.print("T: ");
  lcd.print(temp,0);
  lcd.print("F");

  lcd.setCursor(0, 1);
  lcd.print("H: ");
  lcd.print(humid, 0);
  lcd.print("%");

  lcd.print(" ");

  lcd.print("WL: ");
  lcd.print(water_level, 0);
  lcd.print("%");
  
  delay(2000);
  
}
