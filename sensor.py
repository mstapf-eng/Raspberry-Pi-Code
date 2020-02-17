import time
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import RPi.GPIO as GPIO
import busio
import board
import adafruit_vl53l0x
from mpu6050 import mpu6050
sensor = mpu6050(0x68)
GPIO.setmode(GPIO.BCM)

#setup time of floght sensor so that it can collect readings
#initialize i2c connection
i2c = busio.I2C(board.SCL, board.SDA)
vl53 = adafruit_vl53l0x.VL53L0X(i2c)

#publish message to MQTT
def sendMessage(test, message):
        publish.single(test, payload=message, qos=0, retain=False, hostname="tailor.cloudmqtt.com",
                        port=10720, client_id="de5282f7-c795-420e-87d9-17d0fe059c5c", keepalive=60, 
                        will=None, auth={'username':"xrosxprm", 'password':"i5LvdfdUqJY-"}, tls=None,
                        protocol=mqtt.MQTTv311, transport="tcp")
def callback(topic, channel):
        while GPIO.input(channel) == 1: #channel is on and circuit isn't open
                dist = "Range: {0}mm".format(v153.range)
                print(dist)
                sendMessage(topic, str(dist))
                time.sleep(2) #2 seconds between messages
               
#setup GPIO
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.add_event_detect(17, GPIO.FALLING, callback=my_callback, bouncetime=300)  
try:
        while True:
                time.sleep(1) #1 second between messages
except KeyboardInterrupt:
    GPIO.cleanup()       # clean up GPIO on CTRL+C exit
GPIO.cleanup()           # clean up GPIO on normal exit
