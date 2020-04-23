volatile uint16_t phase = 0;
volatile uint16_t freq = 0;

void setup() {
    pinMode(A0, INPUT);
    timer2_init_pwm();
    timer0_init_ctc();
}

void loop() {
    freq = analogRead(A0);
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
    OCR0A = 250;              // Set counter match value
    sei();                    // Enble interrupts
}

/*
 * Timer0's compare match (channel A) interrupt
 */
ISR(TIMER0_COMPA_vect) {
    phase += freq;
    OCR2A = OCR2B = phase >> 8;
}
