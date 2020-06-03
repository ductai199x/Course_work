/*
 * led_queue.ino
 * Demonstration of queued LED color change events with specified duration
 * 
 * Notes: 
 * - Uses the ArduinoQueue library found in Arduino Library Manager
 */

#include <ArduinoQueue.h>  

// Indicator LED PINs
const int PIN_LED_R = 41;
const int PIN_LED_G = 42;
const int PIN_LED_B = 43;

// Create a custom type to hold an LED color and duration
typedef struct LEDStep {
  char color;
  long dur_ms;
} LEDStep;

// Current LED step
LEDStep current_led_step = {'w', 1000};

// Initialize a FIFO queue to hold LED steps
const int LED_QUEUE_LEN = 16;
ArduinoQueue<LEDStep> led_queue(LED_QUEUE_LEN);

// Timing control
long t0, t1, dt;

// ==============================
void setup() {

  led_init();

  // Add some steps to the queue
  led_queue.enqueue({'r', 1000});
  led_queue.enqueue({'y', 1000});
  led_queue.enqueue({'g', 1000});
  led_queue.enqueue({'c', 1000});
  led_queue.enqueue({'b', 1000});
  led_queue.enqueue({'m', 1000});
  led_queue.enqueue({'w', 1000});
  led_queue.enqueue({'k', 1000});
}

void loop() {
  led_update();
}
// ==============================

// Initialize LED pins; set the first step color and record start time
void led_init() {
  pinMode(PIN_LED_R, OUTPUT);
  pinMode(PIN_LED_G, OUTPUT);
  pinMode(PIN_LED_B, OUTPUT);
  led_set_color(current_led_step.color);
  t0 = millis(); 
}

// Load LED steps from the queue when the current step's duration is up
void led_update() {
  t1 = millis();
  dt = t1 - t0;
  if (dt < current_led_step.dur_ms)
    return;
  if (!led_queue.isEmpty()) {
    current_led_step = led_queue.dequeue();   // Update current step
    led_set_color(current_led_step.color);    // Set the color
    t0 = millis();                            // Record start time
  }
}

// Set one of eight discrete colors ('r', 'y', 'g', 'c', 'b', 'm', 'w', 'k')
// where 'k' turns the LED off
void led_set_color(char color) {
  bool r, g, b;
  r = g = b = false;
  switch (color) {
    case 'r':
      r = true;
      break;
    case 'g':
      g = true;
      break;
    case 'b':
      b = true;
      break;
    case 'c':
      g = b = true;
      break;
    case 'm':
      r = b = true;
      break;
    case 'y':
      r = g = true;
      break;
    case 'w':
      r = g = b = true;
      break;
    case 'k':
    default:
      break;
  }
  digitalWrite(PIN_LED_R, r);
  digitalWrite(PIN_LED_G, g);
  digitalWrite(PIN_LED_B, b);
}
