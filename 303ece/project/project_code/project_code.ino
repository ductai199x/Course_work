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


#include <Servo.h>
#include <scpiparser.h>
#include <Arduino.h>
#include <SPI.h>
#include <MFRC522.h>

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
#define ULT_ECHO 6

#define SERVO_PIN 4

#define WATER_PIN A0
#define THERM_PIN A1
#define IR_RECV_PIN 44

#define RED_LED 23
#define BLUE_LED 25
#define GREEN_LED 27
#define WHITE_LED 3

#define BUZZER 9

volatile unsigned long LastPulseTime;
volatile unsigned long startTime;

int servo_angle = 0;

Servo servo;
MFRC522 mfrc522(RFID_SDA, RFID_RST);

struct scpi_parser_context ctx;

scpi_error_t identify(struct scpi_parser_context* context, struct scpi_token* command);
scpi_error_t get_distance(struct scpi_parser_context* context, struct scpi_token* command);

scpi_error_t set_servo(struct scpi_parser_context* context, struct scpi_token* command);
scpi_error_t set_rgb(struct scpi_parser_context* context, struct scpi_token* command);
scpi_error_t set_speed(struct scpi_parser_context* context, struct scpi_token* command);


void setup() {
  // put your setup code here, to run once:
  Serial.begin (19200);
  servo.attach(SERVO_PIN);
  
  pinMode(ULT_TRIG, OUTPUT);
  pinMode(ULT_ECHO, INPUT);
  attachInterrupt(0, EchoPin_ISR, CHANGE);
  
  pinMode(RED_LED, OUTPUT);
  pinMode(BLUE_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  digitalWrite(RED_LED, LOW);
  digitalWrite(BLUE_LED, LOW);
  digitalWrite(GREEN_LED, LOW);

  scpi_init(&ctx);

  scpi_register_command(ctx.command_tree, SCPI_CL_SAMELEVEL, "*IDN?", 5, "*IDN?", 5, identify);
  
  struct scpi_command* get_distance_cmd;
  get_distance_cmd = scpi_register_command(ctx.command_tree, SCPI_CL_CHILD, "MEASURE", 7, "MEAS", 4, NULL);
  scpi_register_command(get_distance_cmd, SCPI_CL_CHILD, "DISTANCE", 8, "DIST?", 5, get_distance);

//  struct scpi_command* set_servo_cmd;
//  set_servo_cmd = scpi_register_command(ctx.command_tree, SCPI_CL_CHILD, "SET", 3, "SET", 3, NULL);
//  scpi_register_command(set_servo_cmd, SCPI_CL_CHILD, "SERVO", 5, "SERVO", 5, set_servo);

  struct scpi_command* set_rgb_cmd;
  set_rgb_cmd = scpi_register_command(ctx.command_tree, SCPI_CL_CHILD, "SET", 3, "SET", 3, NULL);
  scpi_register_command(set_rgb_cmd, SCPI_CL_CHILD, "RGB", 3, "RGB", 3, set_rgb);

//  struct scpi_command* set_speed_cmd;
//  set_speed_cmd = scpi_register_command(ctx.command_tree, SCPI_CL_CHILD, "SET", 3, "SET", 3, NULL);
//  scpi_register_command(set_speed_cmd, SCPI_CL_CHILD, "SPEED", 5, "SPEED", 5, set_speed);
  
}

void loop()
{
  char line_buffer[256];
  unsigned char read_length;

  while(1)
  {
    /* Read in a line and execute it. */
    read_length = Serial.readBytesUntil('\n', line_buffer, 256);
    if(read_length > 0)
    {
      scpi_execute_command(&ctx, line_buffer, read_length);
    }
  }
}

scpi_error_t identify(struct scpi_parser_context* context, struct scpi_token* command)
{
  scpi_free_tokens(command);
  Serial.println("ECE-303,SimpleDMM,1,10");
  return SCPI_SUCCESS;
}

scpi_error_t get_distance(struct scpi_parser_context* context, struct scpi_token* command)
{
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  Serial.println(LastPulseTime);
  scpi_free_tokens(command);
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

  return SCPI_SUCCESS;
}

void EchoPin_ISR() {
  if (digitalRead(ULT_ECHO)) // Gone HIGH
    startTime = micros();
  else  // Gone LOW
    LastPulseTime = micros() - startTime;
}
