#include "dht.h"


#define DHTPIN 48 // Analog Pin sensor is connected to
#define DHTTYPE DHT11
 
DHT dht(DHTPIN, DHTTYPE, 1);
 
void setup(){
 
  Serial.begin(9600);
  delay(500);//Delay to let system boot
  Serial.println("DHT11 Humidity & temperature Sensor\n\n");
  delay(1000);//Wait before accessing Sensor
 
}//end "setup()"
 
void loop(){
  //Start of Program 
 
//    dht.read(dht_apin);
    int chk = dht.read(DHTPIN);
    Serial.print("Current humidity = ");
    Serial.print(dht.readHumidity());
    Serial.print("%  ");
    Serial.print("temperature = ");
    Serial.print(dht.readTemperature()); 
    Serial.println("C  ");
    
    delay(5000);//Wait 5 seconds before accessing sensor again.
 
  //Fastest should be once every two seconds.
 
}// end loop(
