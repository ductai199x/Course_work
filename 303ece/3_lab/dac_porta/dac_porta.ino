volatile uint16_t phase = 0;
volatile uint16_t freq = 0;

void setup() {
    pinMode(A0, INPUT);
    porta_init();
    timer0_init_ctc();
}

void loop() {
    freq = analogRead(A0);
}

/*
 * Initialize PORTA pins. Equivalent to: 
 *    pinMode(22, OUTPUT);
 *    pinMode(23, OUTPUT);
 *    ...
 *    pinMode(29, OUTPUT);
 */
void porta_init() {
    DDRA = 0xFF;    // Set all 8 of PORTA's pins as outputs (Mega pins 22-29)
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
    OCR0A = 250;              // Set counter match value
    sei();                    // Enble interrupts
}

/*
 * Timer0's compare match (channel A) interrupt
 */
ISR(TIMER0_COMPA_vect) {
    phase += freq;
    PORTA = phase >> 8;
}
