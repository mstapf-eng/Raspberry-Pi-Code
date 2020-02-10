#need to download paho before use
#https://pypi.org/project/paho-mqtt/
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish

#example_topic is topic to subscribe to
#payload is what you want the topic to display
#qos, options either 0,1,2, defaults to 0
#hostname is broker http
#port is the internet port using through cloud broker
#client-id should be populated from client app
#auth are account permissions
#protocol is the mqtt version, in this case 3.1.1

publish.single("example_topic", payload="test", qos=0, retain=False, hostname="tailor.cloudmqtt.com"
        port=10720, client_id="de5282f7-c795-420e-87d9-17d0fe059c5c", keepalive=60, will=None,
        auth={'username':"xrosxprm", 'password':"i5LvdfdUqJY-"}, tls=None,
        protocol=mqtt.MQTTv311, transport="tcp")
