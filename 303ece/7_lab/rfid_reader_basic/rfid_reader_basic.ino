/*
 * rfid_reader_basic.ino
 * Checks if a scanned RFID tag's unique identifier (UID) is authorized and displays result
 * 
 * Notes:
 * - Uses the MFRC522 library for Mifare RC522 Devices.
 * - Based on RFID authorization example at:
 * https://randomnerdtutorials.com/security-access-using-mfrc522-rfid-reader-with-arduino/
 */

#include <SPI.h>        // RC522 Module uses SPI protocol
#include <MFRC522.h>    

// Create MFRC522 instance.
const int SS_PIN = 53;
const int RST_PIN = 5;
MFRC522 mfrc522(SS_PIN, RST_PIN);

// Authorized UID
byte authorized_uid[4] = {0x60, 0x01, 0x86, 0xA6};

void setup() {
  Serial.begin(9600);
  SPI.begin();
  mfrc522.PCD_Init();
}

void loop() {

  // Check for cards
  if (!mfrc522.PICC_IsNewCardPresent())
    return 0;

  // Read the card
  if (!mfrc522.PICC_ReadCardSerial())
    return 0;

  // Check if the UID is authorized
  bool pass = true;
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    Serial.print(mfrc522.uid.uidByte[i], HEX);
    Serial.print("\t");
    if (mfrc522.uid.uidByte[i] != authorized_uid[i])
      pass = false;
  }
  Serial.println();

  // Print results
  if (pass) Serial.println("Access authorized");
  else      Serial.println("Access denied");

//  delay(3000);  // Don't read again for 3 seconds
}
