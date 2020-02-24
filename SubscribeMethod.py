import time
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish

def sendMessage(topic, message):
        publish.single(topic, payload=message, qos=0, retain=False, hostname="$
                        port=10720, client_id="de5282f7-c795-420e-87d9-17d0fe0$
                        protocol=mqtt.MQTTv311, transport="tcp")
while True:
        sendMessage("example_topic","Message") 
        print("Sending Message to MQTT")
        time.sleep(2)
