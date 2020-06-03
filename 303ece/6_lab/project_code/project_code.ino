#include <Servo.h>
#include <scpiparser.h>
#include <Arduino.h>

#define SERVO_PIN 9

volatile unsigned long LastPulseTime;
volatile unsigned long startTime;

#define trigPin 4
#define echoPin 2
#define RED_LED 48
#define BLUE_LED 50
#define GREEN_LED 52

int servo_angle = 0;

Servo servo;

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
  
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
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
    if (digitalRead(echoPin)) // Gone HIGH
        startTime = micros();
    else  // Gone LOW
        LastPulseTime = micros() - startTime;
}
