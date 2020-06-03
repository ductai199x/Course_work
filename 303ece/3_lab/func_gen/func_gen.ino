#include <scpiparser.h>
#include <Arduino.h>

#define TAB_LEN 1024

volatile uint16_t a0;   // ADC ch 0 result
const float F0_MIN = 0.1;
const float F0_MAX = 30;
uint16_t freqtable[TAB_LEN];
uint16_t wavetable[TAB_LEN];

volatile uint16_t phase = 0;
volatile uint16_t freqADC = 0;
volatile uint16_t freqSCPI = 0;
volatile boolean isSCPI = false;
volatile uint16_t sine_sample = 0;

struct scpi_parser_context ctx;

scpi_error_t identify(struct scpi_parser_context* context, struct scpi_token* command);
scpi_error_t identify(struct scpi_parser_context* context, struct scpi_token* command);
scpi_error_t get_distance(struct scpi_parser_context* context, struct scpi_token* command);


void wavetable_init();
void freqtable_init(float f_min, float f_max);
void adc_init_pins();
void adc_init_free_running();
void timer2_init_pwm();
void timer0_init_ctc();

void setup() {
    Serial.begin(19200);
    adc_init_pins();
    adc_init_free_running();
    timer2_init_pwm();
    timer0_init_ctc();
    freqtable_init(F0_MIN, F0_MAX);
    wavetable_init();

    scpi_init(&ctx);

    scpi_register_command(ctx.command_tree, SCPI_CL_SAMELEVEL, "*IDN?", 5, "*IDN?", 5, identify);
    
    struct scpi_command* set_freq_cmd;
    set_freq_cmd = scpi_register_command(ctx.command_tree, SCPI_CL_CHILD, "SOURCE", 6, "SOUR", 4, NULL);
    scpi_register_command(set_freq_cmd, SCPI_CL_CHILD, "FREQUENCY", 9, "FREQ", 4, set_freq);
}

void loop() {
    char line_buffer[256];
    unsigned char read_length;

//    Serial.println(sine_sample);
//    Serial.print(",");
//    Serial.println(a0);
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

/*
 * Respond to *IDN?
 */
scpi_error_t identify(struct scpi_parser_context* context, struct scpi_token* command)
{
  scpi_free_tokens(command);
  Serial.println("ECE-303,SimpleDMM,1,10");
  return SCPI_SUCCESS;
}

scpi_error_t set_freq(struct scpi_parser_context* context, struct scpi_token* command)
{
    struct scpi_token* args;
    struct scpi_numeric output_numeric;
    unsigned char output_value;
  
    args = command;
  
    while (args != NULL && args->type == 0)
    {
        args = args->next;
    }
  
    output_numeric = scpi_parse_numeric(args->value, args->length, F0_MIN, F0_MIN, F0_MAX);
    if (output_numeric.unit[0] != 'H')
    {
        Serial.println("Command error;Invalid unit");
        scpi_error error;
        error.id = -200;
        error.description = "Command error;Invalid unit";
        error.length = 26;
        Serial.print(output_numeric.length);
    
        scpi_queue_error(&ctx, error);
        scpi_free_tokens(command);
        return SCPI_SUCCESS;
    }
    else
    {
        if (output_numeric.value > F0_MAX || output_numeric.value < F0_MIN)
        {
            Serial.println("Command error;Out of range");
            scpi_error error;
            error.id = -201;
            error.description = "Command error;Out of range";
            error.length = 34;
            Serial.print(output_numeric.length);
        
            scpi_queue_error(&ctx, error);
            scpi_free_tokens(command);
            return SCPI_SUCCESS;
        }
        else
        {
            freqSCPI = (uint16_t)output_numeric.value;
            Serial.print("Setting freq to ");
            Serial.print(freqSCPI);
            Serial.println("Hz");
            
            freqSCPI = freqSCPI*(8000/1024);
            isSCPI = true;
        }
    }
    
    scpi_free_tokens(command);

    return SCPI_SUCCESS;
}

void wavetable_init() {
    for (int i = 0; i < TAB_LEN; i++) {
        wavetable[i] = (1 + sinf(2*M_PI*i / (TAB_LEN-1))) * 32767.5;
    }
}


// Exponential mapping between freq and pot ADC reading
void freqtable_init(float f_min, float f_max) {
   float coeff = powf(f_max/f_min, 1.0/(TAB_LEN-1));
   float val = f_min;
   for (int i = 0; i < TAB_LEN; i++) {
       freqtable[i] = roundf(val * 65535.5);
       val *= coeff;
   }
}


// Linear mapping between freq and pot ADC reading
void freqtable_init(float f_min, float f_max) {
    for (int i = 0; i < TAB_LEN; i++) {
        freqtable[i] = roundf(i*(f_max-f_min)/1024);
    }
}

/*
 * More or less equivalent to pinMode(A0, INPUT). Also disables digital input
 * buffer on A0 to save power
 */
void adc_init_pins() {
    DDRF &= ~(1 << PF0);      // Configure first pin of PORTF as input
    DIDR0 |= (1 << ADC0D);    // Disable digital input buffer
}

/*
 * Initialize the ADC for free running mode with prescaler = 128, yielding
 * ADC clock freq 16e6/128 = 125kHz (need clock between 50kHz and 200kHz for
 * 10-bit resolution), and sample rate fs = 16e6/128/13=9.615kHz
 */
void adc_init_free_running() {
    ADMUX |= (1 << REFS0);    // Use AVCC as the reference
    ADMUX &= ~(1 << ADLAR);   // Right-adjust conversion results
    ADCSRA |= (1 << ADEN);    // Enable ADC
    ADCSRA |= (1 << ADIE);    // Call ISR(ADC_vect) when conversion is complete
    ADCSRA |= (1 << ADATE);                                   // Enabble auto-trigger
    ADCSRB &= ~((1 << ADTS2) | (1 << ADTS1) | (1 << ADTS0));  // Free running mode
    ADCSRA |= (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);     // Set prescaler 
    ADCSRA |= (1 << ADSC);    // Start first conversion
    sei();                    // Enable interrupts
}

/*
 * ADC interrupt service routine. Enabled by ADIE bit in ADCSRA register.
 * 
 * Note: if we needed to read multiple ADC channels, we would read one per ISR
 * by reading the MUX bits (MUX4:0 in ADMUX and the MUX5 bit in ADCSRB) to get the 
 * channel of the most recent conversion, store the result accordingly, and 
 * set the MUX bits to read from the next channel we want.
 */
ISR(ADC_vect) {
    uint8_t lsb = ADCL;
    uint8_t msb = ADCH;
    a0 = (msb << 8) | lsb;
    freqADC = freqtable[a0]*(8000/1024);
}


/* 
 *  Initialize Timer2 for fast PWM mode, at a frequency of 16e6/1/256 = 62.5kHz
 */
void timer2_init_pwm() {
    DDRB |= (1 << PB4); // Enable output on channel A (PB4, Mega pin 10)
    DDRH |= (1 << PH6); // Enable output on channel B (PH6, Mega pin 9)
    TCCR2A = 0;         // Clear control register A
    TCCR2B = 0;         // Clear control register B
    TCCR2A |= (1 << WGM21) | (1 << WGM20);  // Fast PWM (mode 3)
    TCCR2A |= (1 << COM2A1);  // Non-inverting mode (channel A)
    TCCR2A |= (1 << COM2B1);  // Non-inverting mode (channel B)
    TCCR2B |= (1 << CS20);    // Prescaler = 1
}

/*
 * Initialize Timer0 to control audio sample timing at fs = 8kHz
 */
void timer0_init_ctc() {
    TCCR0A = 0;               // Clear control register A
    TCCR0B = 0;               // Clear control register B
    TCCR0A |= (1 << WGM01);   // CTC (mode 2)
    TIMSK0 |= (1 << OCIE2A);  // Interrupt on OCR0A
    TCCR0B |= (1 << CS01);    // Prescaler = 8
    cli();                    // Disable interrupts
    TCNT0 = 0;                // Initialize counter
    OCR0A = 125;              // Set counter match value
    sei();                    // Enble interrupts
}

/*
 * Timer0's compare match (channel A) interrupt
 */
ISR(TIMER0_COMPA_vect) {
    phase += isSCPI ? freqSCPI : freqADC;
    sine_sample = wavetable[phase >> 6];
    OCR2A = OCR2B = (sine_sample/256);
}
