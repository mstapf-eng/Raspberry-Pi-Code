import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish

publish.single(topic, payload=None, qos=0, retain=False, hostname="tailor.cloudmqtt.com",
    port=10720, client_id="de5282f7-c795-420e-87d9-17d0fe059c5c", keepalive=60, will=None, auth={'username':"xrosxprm", 'password':"i5LvdfdUqJY-"}, tls=None,
    protocol=mqtt.MQTTv311, transport="tcp")

msgs = [{'topic':"paho/test/multiple", 'payload':"multiple 1"},
    ("paho/test/multiple", "multiple 2", 0, False)]

publish.multiple(msgs, hostname="tailor.cloudmqtt.com", port=10720, client_id="de5282f7-c795-420e-87d9-17d0fe059c5c", keepalive=60,
    will=None, auth={'username':"xrosxprm", 'password':"i5LvdfdUqJY-"}, tls=None, protocol=mqtt.MQTTv311, transport="tcp")
