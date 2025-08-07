from itertools import chain
from ssl import PROTOCOL_TLS
from typing import Callable, Sequence, Dict

import paho.mqtt.client as mqtt
import numpy as np
import json

from ascribe_link.example import sphere_example

# Define the enumerated functions for data processing
def random_mesh(*args, **kwargs):
    # Implement processing logic here
    return np.random.rand(10, 3)

# Define a dictionary mapping function names to implementations
function_map = {
    'sphere': sphere_example,
    'random_mesh': random_mesh
}

def validate_mesh(points, indices):
    # Check for non-finite vertices
    bad_points = np.array(points)[~np.isfinite(points).all(axis=1)]
    if len(bad_points):
        print("Bad points:", bad_points)
        raise ValueError("Mesh contains points with infinite values")

# Define a callback for incoming processing requests
def on_message(client, userdata, message):
    topic = message.topic
    request_data = json.loads(message.payload)
    print(client, userdata, topic, request_data)
    match topic:
        case 'godot/processing_requests':
            function_name = request_data['function_name']
            args = request_data['args']
            kwargs = request_data['kwargs']

            # Call the corresponding function and serialize the result
            result = function_map[function_name](*args, **kwargs)
            result_data = {'vertices': list(chain.from_iterable(result[0])),
                           'indices': result[1]}

            # Validate before sending
            validate_mesh(result[0], result[1])

            # Publish the result to the processing responses topic
            client.publish("python/processing_responses", json.dumps(result_data))
        case 'godot/specimen_requests':
            function_names = list(function_map.keys())
            client.publish("python/specimen_responses", json.dumps(function_names))5

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected with result code {reason_code}")
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("$SYS/#")

def serve(broker=None, port=1883, client=None, mesh_functions: Dict[str, Callable]=None):
    if mesh_functions:
        function_map.clear()
        function_map.update(mesh_functions)

    if client is None:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect(broker, port)

    # Subscribe to the processing requests topic
    client.subscribe("godot/processing_requests")

    # Subscribe to the specimen requests topic
    client.subscribe("godot/specimen_requests")

    # Set the callback for incoming messages
    client.on_message = on_message
    client.on_connect = on_connect

    # Start the MQTT loop
    client.loop_forever()


if __name__ == '__main__':
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)#protocol=mqtt.MQTTv5)#, transport="websockets")
    client.tls_set(tls_version=PROTOCOL_TLS)
    # client.ws_set_options(path="/mqtt")
    client.username_pw_set("ascribe", "Ascribe1")
    client.connect("1d7af061725546779afb0f88f1577d45.s1.eu.hivemq.cloud", 8883)
    serve(client=client)
