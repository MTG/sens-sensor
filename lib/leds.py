# Function to turn LEDs on
def turn_leds_on(GPIO,led_pins):
    for pin in led_pins:
        GPIO.output(pin, GPIO.HIGH)

# Function to turn LEDs off
def turn_leds_off(GPIO,led_pins):
    for pin in led_pins:
        GPIO.output(pin, GPIO.LOW)