volatile uint16_t a0;   // ADC ch 0 result

void setup() {
    Serial.begin(9600);
    adc_init_pins();
    adc_init_free_running();
}

void loop() {
    Serial.println(a0);
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
}
