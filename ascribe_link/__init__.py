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
    return np.random.rand(10, 3), np.random.randint(10, (10, 3))

# Define a dictionary mapping function names to implementations
function_map = {
    'sphere': sphere_example,
    'random_mesh': random_mesh
}




def type_to_schema(annotation):
    # base case
    if annotation is inspect._empty:
        return {}
    # get the type and the arguments
    origin = get_origin(annotation)
    args = get_args(annotation)

    # if it's an annotated type
    if origin is Annotated:
        base_type = args[0]
        metadata_items = args[1:]
        # recurse to the next part of the annotation
        base_schema = type_to_schema(base_type)

        # Apply known metadata types
        for metadata in metadata_items:
            metadata_class_name = metadata.__class__.__name__
            
        return base_schema
    
    if origin is Literal:
        literal_values = list(args)
        schema_fragment: dict[str, Any] = {"enum": literal_values}
        print(origin)
        print(args)

        # handle literals in the case of them being in tuples or some other complicated way
        if literal_values and all(isinstance(value, str) for value in literal_values):
            schema_fragment["type"] = "string"
        elif literal_values and all(isinstance(value, bool) for value in literal_values):
            schema_fragment["type"] = "boolean"
        elif literal_values and all(isinstance(value, (int, float)) and not isinstance(v, bool) for value in literal_values):
            schema_fragment["type"] = "number"
        elif literal_values and literal_values is tuple:
            schema_fragment["type"] = "string"
            schema_fragment["enum"] = list(literal_values)
        return schema_fragment

     # basic python types
    if annotation is str:
        return {"type": "string"}
    if annotation is int or annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is type(None):
        return {"type": "null"}

    # handle unknown types
    return {}
    

def create_schema(func: Callable):

    resolved_type_hints = get_type_hints(
        func,
        globalns=getattr(func, "__globals__", None),
        localns=None,
        include_extras=True,
    )
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.com/person.schema.json",
        "title": func.__name__,
        "type": "object",
        "properties": None
        
    }
    # set the name of the function
    schema['title'] = func.__name__

    # based on the signature we can figure out the properties and type
    signature = inspect.signature(func)
    

    parameters = {}
    # loop over parameters to get their types and then set them as the properties
    for param_name, param in signature.parameters.items():
        resolved_annotation = resolved_type_hints.get(param_name, param.annotation)
        parameter_schema = type_to_schema(resolved_annotation)
        # print(param)
        parameters[param_name] = parameter_schema
    schema['properties'] = parameters
        

    # validate the schema
    Draft202012Validator.check_schema(schema)
    return schema

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
            response = dict(names=function_names)
            client.publish("python/specimen_responses", json.dumps(response))

        case 'godot/function_schemas':
            function_name = request_data['function_name']
            func = function_map[function_name]
            schema = create_schema(func)
            client.publish("python/function_schemas", schema)
            
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
    # Subscribe to function schemas topic
    client.subscribe("godot/function_schemas")

    # Set the callback for incoming messages
    client.on_message = on_message
    client.on_connect = on_connect

    # Start the MQTT loop
    client.loop_forever()




if __name__ == '__main__':
    # Client setup
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # , transport="websockets")
    client.connect("vision.lbl.gov", 1883)

    # Start server
    serve(client=client, mesh_functions={"Automated Thresholding":0, "Unsupervised ML":1, "Supervised ML":2})
