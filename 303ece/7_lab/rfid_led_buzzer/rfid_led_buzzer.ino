/*
 * Typical pin layout used:
 * -----------------------------------------------------------------------------------------
 *             MFRC522      Arduino       Arduino   Arduino    Arduino          Arduino
 *             Reader/PCD   Uno/101       Mega      Nano v3    Leonardo/Micro   Pro Micro
 * Signal      Pin          Pin           Pin       Pin        Pin              Pin
 * -----------------------------------------------------------------------------------------
 * RST/Reset   RST          9             5         D9         RESET/ICSP-5     RST
 * SPI SS      SDA(SS)      10            53        D10        10               10
 * SPI MOSI    MOSI         11 / ICSP-4   51        D11        ICSP-4           16
 * SPI MISO    MISO         12 / ICSP-1   50        D12        ICSP-1           14
 * SPI SCK     SCK          13 / ICSP-3   52        D13        ICSP-3           15
 */
 
#include <SPI.h>
#include <MFRC522.h>

#include "pitches.h"

// Indicator LED PINs
#define PIN_LED_R 43
#define PIN_LED_G 42
#define PIN_LED_B 41

#define BUZZER 8

#define SS_PIN 53
#define RST_PIN 5
MFRC522 mfrc522(SS_PIN, RST_PIN);   // Create MFRC522 instance.

boolean is_allow_to_scan = true;
unsigned long rfid_scan_timer = 0;
const int rfid_scan_cooldown = 3000;

unsigned long timer_led0 = 0;
unsigned long timer_led1 = 0;
const int led_duration = 500; // milliseconds
int current_led_state = 0;

boolean buzzer_on = 0;
int HIGH_NOTE = NOTE_A5;
int LOW_NOTE = NOTE_A1;

boolean is_authorized = false;
 
void setup() 
{
  pinMode(PIN_LED_R, OUTPUT);
  pinMode(PIN_LED_G, OUTPUT);
  pinMode(PIN_LED_B, OUTPUT);
  digitalWrite(PIN_LED_R, LOW);
  digitalWrite(PIN_LED_G, HIGH);
  digitalWrite(PIN_LED_B, LOW);
  Serial.begin(9600);   // Initiate a serial communication
  SPI.begin();      // Initiate  SPI bus
  mfrc522.PCD_Init();   // Initiate MFRC522
  Serial.println("Ready to scan card...");
}

void loop() 
{
  if (!is_allow_to_scan) {
    if (millis() - rfid_scan_timer > rfid_scan_cooldown) {
      current_led_state = 0;
      digitalWrite(PIN_LED_R, LOW);
      digitalWrite(PIN_LED_G, HIGH);
      digitalWrite(PIN_LED_B, LOW);
      is_allow_to_scan = true;
      buzzer_on = false;
      noTone(BUZZER);
      Serial.println("Ready to scan card...");
      return;
    }
    
    if (is_authorized && !buzzer_on)
      tone(BUZZER, HIGH_NOTE, rfid_scan_cooldown);
    if (!is_authorized && !buzzer_on)
      tone(BUZZER, LOW_NOTE, rfid_scan_cooldown);

    flash_LEDS();
    
    buzzer_on = true;
    return;
  }
  
  // Look for new cards
  if ( ! mfrc522.PICC_IsNewCardPresent()) 
  {
    return;
  }
  // Select one of the cards
  if ( ! mfrc522.PICC_ReadCardSerial()) 
  {
    return;
  }
  //Show UID on serial monitor
  Serial.print("UID tag :");
  String content= "";
  byte letter;
  for (byte i = 0; i < mfrc522.uid.size; i++) 
  {
     Serial.print(mfrc522.uid.uidByte[i] < 0x10 ? " 0" : " ");
     Serial.print(mfrc522.uid.uidByte[i], HEX);
     content.concat(String(mfrc522.uid.uidByte[i] < 0x10 ? " 0" : " "));
     content.concat(String(mfrc522.uid.uidByte[i], HEX));
  }
  Serial.println();
  Serial.print("Message : ");
  content.toUpperCase();

  if (content.substring(1) == "E2 A6 D2 1B") //change here the UID of the card/cards that you want to give access
  {
    Serial.println("Authorized access");
    is_authorized = true;
  } else {
    Serial.println("Access denied");
    is_authorized = false;
  }
  is_allow_to_scan = false;
  rfid_scan_timer = millis();
}

void flash_LEDS()
{
  if (is_authorized) {
    switch(current_led_state) {
      case 0:
        current_led_state = 1;
        timer_led0 = millis();
        break;
      case 1:
        // turn on green
        analogWrite(PIN_LED_R, 0);
        analogWrite(PIN_LED_G, 255);
        analogWrite(PIN_LED_B, 0);
        if (millis() - timer_led0 > led_duration) {
          timer_led1 = millis();
          current_led_state = 2;
        }
        break;
      case 2:
        // turn on yellow
        analogWrite(PIN_LED_R, 255);
        analogWrite(PIN_LED_G, 255);
        analogWrite(PIN_LED_B, 0);
        if (millis() - timer_led1 > led_duration) {
          timer_led0 = millis();
          current_led_state = 1;
        }
        break;
    }
  } else {
    switch(current_led_state) {
      case 0:
        current_led_state = 1;
        timer_led0 = millis();
        break;
      case 1:
        // turn on green
        analogWrite(PIN_LED_R, 0);
        analogWrite(PIN_LED_G, 255);
        analogWrite(PIN_LED_B, 0);
        if (millis() - timer_led0 > led_duration) {
          timer_led1 = millis();
          current_led_state = 2;
        }
        break;
      case 2:
        // turn on red
        analogWrite(PIN_LED_R, 255);
        analogWrite(PIN_LED_G, 0);
        analogWrite(PIN_LED_B, 0);
        if (millis() - timer_led1 > led_duration) {
          timer_led0 = millis();
          current_led_state = 1;
        }
        break;
    }
  }
}
