// Timer 0: 13(8) & 4(8)
// Timer 1: 11(16) & 12(16)
// Timer 2: 10(8) & 9(8)
// Timer 3: 5,2,3 (all 16)
// Timer 4: 6,7,8 (all 16)
// Timer 5: 44,45,46 (all 16)


/* LCD:       45=en, 43=rs, 41=d4, 39=d5, 37=d6, 35=d7
 * LEDs:      27=g, 25=b, 23=r, 3=w(PWM)
 * RFID:      53=sda, 52=sck, 51=mosi, 50=miso, 49=rst
 * DC MOTOR:  7, 8 (both PWMs)
 * ULT_SON:   5=trig, 6=echo
 * SERVO:     4
 * WATER:     A0
 * THERM:     A1
 * IR-RECV:   44
 * BUZZER:    9
 */

#define IR_SMALLD_NEC

#include <Servo.h>
#include <scpiparser.h>
#include <Arduino.h>
#include <SPI.h>
#include <MFRC522.h>
#include <IRsmallDecoder.h>
#include "pitches.h"
#include <LiquidCrystal.h>

#define LCD_EN 45
#define LCD_RS 43
#define LCD_D4 41
#define LCD_D5 39
#define LCD_D6 37
#define LCD_D7 35

#define RFID_SDA 53
#define RFID_RST 49

#define DC_PWM1 7
#define DC_PWM2 8

#define ULT_TRIG 5
#define ULT_ECHO 2

#define SERVO_PIN 4

#define WATER_PIN A0
#define THERM_PIN A1
#define IR_RECV_PIN 18

#define RED_LED 23
#define BLUE_LED 25
#define GREEN_LED 27
#define HEADLIGHT_LED 3
#define CTRL_MODE_LED 36

#define BUZZER 12

#define AUTO 0
#define MANUAL 1

#define HEADLIGHT_OFF 22
#define HEADLIGHT_DIM 12
#define HEADLIGHT_ON 24
#define SWITCH_CTRL 69

#define DEC_SPEED 7
#define INC_SPEED 9

volatile unsigned long LastPulseTime = 10000;
volatile unsigned long startTime;

int servo_angle = 0;
int control_mode = 0;

boolean is_allow_to_scan = true;
unsigned long rfid_scan_timer = 0;
const int rfid_scan_cooldown = 3000;

unsigned long lcd_update_timer = 0;
const int lcd_update_interval = 600;

boolean buzzer_on = 0;
int HIGH_NOTE = NOTE_A6;
int LOW_NOTE = NOTE_A4;
boolean is_authorized = false;

int motor_speed = 0;
int headlight_state = 0;

float temp = 0.0f;
float water_level = 0.0f;
const float R1 = 10000;
float logR2, R2;
const float c1 = 1.009249522e-03, c2 = 2.378405444e-04, c3 = 2.019202697e-07;


Servo servo;
MFRC522 mfrc522(RFID_SDA, RFID_RST);
IRsmallDecoder irrecv(IR_RECV_PIN);
irSmallD_t irData;
LiquidCrystal lcd(LCD_RS, LCD_EN, LCD_D4, LCD_D5, LCD_D6, LCD_D7);

struct scpi_parser_context ctx;

scpi_error_t identify(struct scpi_parser_context* context, struct scpi_token* command);
scpi_error_t get_all_sensors(struct scpi_parser_context* context, struct scpi_token* command);
scpi_error_t get_distance(struct scpi_parser_context* context, struct scpi_token* command);

scpi_error_t get_ctrl_mode(struct scpi_parser_context* context, struct scpi_token* command);
scpi_error_t get_headlight_state(struct scpi_parser_context* context, struct scpi_token* command);
scpi_error_t get_speed(struct scpi_parser_context* context, struct scpi_token* command);
scpi_error_t get_all_states(struct scpi_parser_context* context, struct scpi_token* command);

scpi_error_t set_servo(struct scpi_parser_context* context, struct scpi_token* command);
scpi_error_t set_rgb(struct scpi_parser_context* context, struct scpi_token* command);
scpi_error_t set_speed(struct scpi_parser_context* context, struct scpi_token* command);

scpi_error_t set_all(struct scpi_parser_context* context, struct scpi_token* command);


void setup() {
  // put your setup code here, to run once:
  Serial.begin (9600);
  servo.attach(SERVO_PIN);
  
  pinMode(ULT_TRIG, OUTPUT);
  pinMode(ULT_ECHO, INPUT);
  attachInterrupt(0, EchoPin_ISR, CHANGE);
  
  pinMode(RED_LED, OUTPUT);
  pinMode(BLUE_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(CTRL_MODE_LED, OUTPUT);
  digitalWrite(RED_LED, LOW);
  digitalWrite(BLUE_LED, LOW);
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(CTRL_MODE_LED, LOW);

  pinMode(DC_PWM1, OUTPUT);
  pinMode(DC_PWM2, OUTPUT);
  analogWrite(DC_PWM1, 0);
  analogWrite(DC_PWM2, 0);

  pinMode(BUZZER, OUTPUT);
  analogWrite(BUZZER, 0);

  pinMode(THERM_PIN, INPUT);
  pinMode(WATER_PIN, INPUT);

  lcd.begin(16, 2);

  SPI.begin();      // Initiate  SPI bus
  mfrc522.PCD_Init();   // Initiate MFRC522

  scpi_init(&ctx);

  scpi_register_command(ctx.command_tree, SCPI_CL_SAMELEVEL, "*IDN?", 5, "*IDN?", 5, identify);

  struct scpi_command* get_all_sensors_cmd;
  get_all_sensors_cmd = scpi_register_command(ctx.command_tree, SCPI_CL_CHILD, "MEAS", 4, "MEAS", 4, NULL);
  scpi_register_command(get_all_sensors_cmd, SCPI_CL_CHILD, "ALL", 3, "ALL", 3, get_all_sensors);
  
  struct scpi_command* get_distance_cmd;
  get_distance_cmd = scpi_register_command(ctx.command_tree, SCPI_CL_CHILD, "MEAS0", 5, "MEAS0", 5, NULL);
  scpi_register_command(get_distance_cmd, SCPI_CL_CHILD, "DISTANCE", 8, "DIST?", 5, get_distance);

  struct scpi_command* get_all_states_cmd;
  get_all_states_cmd = scpi_register_command(ctx.command_tree, SCPI_CL_CHILD, "GET", 3, "GET", 3, NULL);
  scpi_register_command(get_all_states_cmd, SCPI_CL_CHILD, "ALL", 3, "ALL", 3, get_all_states);

  struct scpi_command* get_ctrl_mode_cmd;
  get_ctrl_mode_cmd = scpi_register_command(ctx.command_tree, SCPI_CL_CHILD, "GET0", 4, "GET0", 4, NULL);
  scpi_register_command(get_ctrl_mode_cmd, SCPI_CL_CHILD, "CTRL_MODE", 9, "CTRL_MODE", 9, get_ctrl_mode);

  struct scpi_command* get_headlight_cmd;
  get_headlight_cmd = scpi_register_command(ctx.command_tree, SCPI_CL_CHILD, "GET1", 4, "GET1", 4, NULL);
  scpi_register_command(get_headlight_cmd, SCPI_CL_CHILD, "HEADLIGHT", 9, "HEADLIGHT", 9, get_headlight_state);

  struct scpi_command* get_speed_cmd;
  get_speed_cmd = scpi_register_command(ctx.command_tree, SCPI_CL_CHILD, "GET2", 4, "GET2", 4, NULL);
  scpi_register_command(get_speed_cmd, SCPI_CL_CHILD, "SPEED", 5, "SPEED", 5, get_speed);

  struct scpi_command* set_all_cmd;
  set_all_cmd = scpi_register_command(ctx.command_tree, SCPI_CL_CHILD, "SET", 3, "SET", 3, NULL);
  scpi_register_command(set_all_cmd, SCPI_CL_CHILD, "ALL", 3, "ALL", 3, set_all);

  struct scpi_command* set_servo_cmd;
  set_servo_cmd = scpi_register_command(ctx.command_tree, SCPI_CL_CHILD, "SET0", 4, "SET0", 4, NULL);
  scpi_register_command(set_servo_cmd, SCPI_CL_CHILD, "SERVO", 5, "SERVO", 5, set_servo);

  struct scpi_command* set_rgb_cmd;
  set_rgb_cmd = scpi_register_command(ctx.command_tree, SCPI_CL_CHILD, "SET1", 4, "SET1", 4, NULL);
  scpi_register_command(set_rgb_cmd, SCPI_CL_CHILD, "RGB", 3, "RGB", 3, set_rgb);

  struct scpi_command* set_speed_cmd;
  set_speed_cmd = scpi_register_command(ctx.command_tree, SCPI_CL_CHILD, "SET2", 4, "SET2", 4, NULL);
  scpi_register_command(set_speed_cmd, SCPI_CL_CHILD, "SPEED", 5, "SPEED", 5, set_speed);
  
}

void loop()
{
  lcd_update();

  sensors();
  
  char line_buffer[256];
  unsigned char read_length;

  if (Serial.available() > 0)
  {
    /* Read in a line and execute it. */
    read_length = Serial.readBytesUntil('\n', line_buffer, 256);
    if(read_length > 0)
    {
      scpi_execute_command(&ctx, line_buffer, read_length);
    }
  }

  ir_remote();

  rfid();
  
}


void lcd_update()
{
  // Update LCD:
  if (millis() - lcd_update_timer > lcd_update_interval) {
    lcd_update_timer = millis();
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("T=");
    lcd.print(temp,0);
    lcd.print("F");
    lcd.setCursor(0, 1);
    lcd.print("MTR=");
    int disp_ms = motor_speed*100/255;
    char val_tmp[3];
    itoa(disp_ms, val_tmp, 10);
//    Serial.println(disp_ms);
    lcd.print(val_tmp);
    lcd.print("%");
    lcd.print(" ");
    lcd.print("WL=");
    lcd.print(water_level, 0);
    lcd.print("%");
  }
}

void sensors()
{
  int V1 = analogRead(THERM_PIN);
  R2 = R1 * (1023.0 / (float)V1 - 1.0);
  logR2 = log(R2);
  temp = (1.0 / (c1 + c2*logR2 + c3*logR2*logR2*logR2));
  temp = temp - 273.15;
  temp = (temp * 9.0)/ 5.0 + 32.0 - 8.0; 

  int V0 = analogRead(WATER_PIN);
  water_level = (float)V0/1024*100;

  digitalWrite(ULT_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(ULT_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULT_TRIG, LOW);
}

void ir_remote()
{
  if (irrecv.dataAvailable(irData)){
//    Serial.println(irData.cmd);
    switch(irData.cmd) {
      case HEADLIGHT_OFF:
        headlight_state = 0;
        analogWrite(HEADLIGHT_LED, 0);
        break;
      case HEADLIGHT_DIM:
        headlight_state = 1;
        analogWrite(HEADLIGHT_LED, 50);
        break;
      case HEADLIGHT_ON:
        headlight_state = 2;
        analogWrite(HEADLIGHT_LED, 255);
        break;
      case SWITCH_CTRL:
        control_mode = AUTO;
        is_authorized = false;
        digitalWrite(CTRL_MODE_LED, LOW);
        tone(BUZZER, LOW_NOTE, 250);
        break;
      case DEC_SPEED:
        if (control_mode == MANUAL) {
          if (motor_speed > 10)
            motor_speed -= 10;
          else
            motor_speed = 0;
          analogWrite(DC_PWM2, motor_speed);
        }
        break;
      case INC_SPEED:
        if (control_mode == MANUAL) {
          if (motor_speed < 245) 
            motor_speed += 10;
          else
            motor_speed = 255;
          analogWrite(DC_PWM2, motor_speed);
        }
        break;
      default:
        break;
    }
  }
}


void rfid()
{
  if (!is_allow_to_scan) {
    if (millis() - rfid_scan_timer > rfid_scan_cooldown) {
      is_allow_to_scan = true;
      buzzer_on = false;
      noTone(BUZZER);
      analogWrite(BUZZER, 0);
      return;
    }
    
    if (is_authorized && !buzzer_on) {
      control_mode = MANUAL;
      digitalWrite(CTRL_MODE_LED, HIGH);
      tone(BUZZER, HIGH_NOTE, 250);
    }
    if (!is_authorized && !buzzer_on) {
      control_mode = AUTO;
      digitalWrite(CTRL_MODE_LED, LOW);
      tone(BUZZER, LOW_NOTE, 250);
    }
    
    buzzer_on = true;
    return;
  }
  
  // Look for new cards
  if (! mfrc522.PICC_IsNewCardPresent()) 
  {
    return;
  }
  // Select one of the cards
  if (! mfrc522.PICC_ReadCardSerial()) 
  {
    return;
  }
  String content= "";
  byte letter;
  for (byte i = 0; i < mfrc522.uid.size; i++) 
  {
     content.concat(String(mfrc522.uid.uidByte[i] < 0x10 ? " 0" : " "));
     content.concat(String(mfrc522.uid.uidByte[i], HEX));
  }
  content.toUpperCase();

  if (content.substring(1) == "E2 A6 D2 1B") //change here the UID of the card/cards that you want to give access
  {
    is_authorized = true;
  } else {
    is_authorized = false;
  }
  is_allow_to_scan = false;
  rfid_scan_timer = millis();
}

scpi_error_t identify(struct scpi_parser_context* context, struct scpi_token* command)
{
  scpi_free_tokens(command);
  Serial.println("ECE-303,SimpleDMM,1,10");
  return SCPI_SUCCESS;
}

scpi_error_t get_all_sensors(struct scpi_parser_context* context, struct scpi_token* command)
{
  String content= "";
  content.concat(String(LastPulseTime));
  content.concat(" ");
  content.concat(String(temp));
  content.concat(" ");
  content.concat(String(water_level));
  scpi_free_tokens(command);
  Serial.println(content);
  return SCPI_SUCCESS;
}

scpi_error_t get_all_states(struct scpi_parser_context* context, struct scpi_token* command)
{
  String content= "";
  content.concat(String(control_mode));
  content.concat(" ");
  content.concat(String(motor_speed));
  content.concat(" ");
  content.concat(String(headlight_state));
  scpi_free_tokens(command);
  Serial.println(content);
  return SCPI_SUCCESS;
}

scpi_error_t get_distance(struct scpi_parser_context* context, struct scpi_token* command)
{
  digitalWrite(ULT_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(ULT_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULT_TRIG, LOW);
  Serial.println(LastPulseTime);
  scpi_free_tokens(command);
  return SCPI_SUCCESS;
}

scpi_error_t get_ctrl_mode(struct scpi_parser_context* context, struct scpi_token* command)
{
  scpi_free_tokens(command);
  Serial.println(control_mode);
  return SCPI_SUCCESS;
}

scpi_error_t get_headlight_state(struct scpi_parser_context* context, struct scpi_token* command)
{
  scpi_free_tokens(command);
  Serial.println(headlight_state);
  return SCPI_SUCCESS;
}

scpi_error_t get_speed(struct scpi_parser_context* context, struct scpi_token* command)
{
  scpi_free_tokens(command);
  Serial.println(motor_speed);
  return SCPI_SUCCESS;
}

scpi_error_t set_servo(struct scpi_parser_context* context, struct scpi_token* command)
{
  struct scpi_token* args;
  struct scpi_numeric output_numeric;

  args = command;

  while (args != NULL && args->type == 0)
  {
      args = args->next;
  }

  output_numeric = scpi_parse_numeric(args->value, args->length, 0, 0, 180);
  servo.write(output_numeric.value);
  delayMicroseconds(10);
  scpi_free_tokens(command);
  return SCPI_SUCCESS;
}


scpi_error_t set_rgb(struct scpi_parser_context* context, struct scpi_token* command)
{
  struct scpi_token* args;

  char light[2];

  args = command;
  while (args != NULL && args->type == 0)
  {
      args = args->next;
  }

  light[0] = args->value[0];
  light[1] = '\0';

  if (light[0] == 'R' || light[0] == 'r') {
    digitalWrite(RED_LED, HIGH);
    digitalWrite(BLUE_LED, LOW);
    digitalWrite(GREEN_LED, LOW);
  } else if (light[0] == 'G' || light[0] == 'g') {
    digitalWrite(RED_LED, LOW);
    digitalWrite(BLUE_LED, LOW);
    digitalWrite(GREEN_LED, HIGH);
  } else if (light[0] == 'B' || light[0] == 'b') {
    digitalWrite(RED_LED, LOW);
    digitalWrite(BLUE_LED, HIGH);
    digitalWrite(GREEN_LED, LOW);
  } else {
    digitalWrite(RED_LED, LOW);
    digitalWrite(BLUE_LED, LOW);
    digitalWrite(GREEN_LED, LOW);
  }
  
  scpi_free_tokens(command);
  return SCPI_SUCCESS;
}


scpi_error_t set_speed(struct scpi_parser_context* context, struct scpi_token* command)
{
  struct scpi_token* args;
  struct scpi_numeric output_numeric;
  unsigned char output_value;

  args = command;

  while(args != NULL && args->type == 0)
  {
    args = args->next;
  }

  output_numeric = scpi_parse_numeric(args->value, args->length, 0, 0, 5);

  motor_speed = (int)((float)output_numeric.value/100*255);
  
  Serial.print("Setting motor speed to: ");
  Serial.println(motor_speed);

  analogWrite(DC_PWM2, motor_speed);
  
  scpi_free_tokens(command);
  return SCPI_SUCCESS;
}

scpi_error_t set_all(struct scpi_parser_context* context, struct scpi_token* command)
{
  struct scpi_token* args;
  args = command;
  while (args != NULL && args->type == 0)
  {
      args = args->next;
  }

  Serial.println(args->value);

  char* cmd = strtok(args->value, ",");
  int i = 0;
  int val = 0;
  while (cmd != 0)
  {
    val = atoi(cmd);
    switch(i) {
      case 0:
        if (val == 0) {
          digitalWrite(RED_LED, HIGH);
          digitalWrite(BLUE_LED, LOW);
          digitalWrite(GREEN_LED, LOW);
        } else if (val == 1) {
          digitalWrite(RED_LED, LOW);
          digitalWrite(BLUE_LED, HIGH);
          digitalWrite(GREEN_LED, LOW);
        } else if (val == 2) {
          digitalWrite(RED_LED, LOW);
          digitalWrite(BLUE_LED, LOW);
          digitalWrite(GREEN_LED, HIGH);
        } else {
          digitalWrite(RED_LED, LOW);
          digitalWrite(BLUE_LED, LOW);
          digitalWrite(GREEN_LED, LOW);
        }
        break;
      case 1:
        servo.write(val);
        delayMicroseconds(10);
        break;
      case 2:
        if (control_mode == MANUAL)
          break;
        motor_speed = (int)((float)val/100*255);
        analogWrite(DC_PWM2, motor_speed);
        break;
    }
    i++;
    // Find the next command in input string
    cmd = strtok(0, ",");
  }
  
  scpi_free_tokens(command);
  return SCPI_SUCCESS;
}


void EchoPin_ISR() {
  if (digitalRead(ULT_ECHO)) // Gone HIGH
    startTime = micros();
  else  // Gone LOW
    LastPulseTime = micros() - startTime;
}
