from image_to_json import image_to_json
from json_to_csv import json_to_csv
from dotenv import load_dotenv
import base64

def image_processor(image_input) -> str:
    """"Receives image, processes it to extract the table data, and returns a CSV string."""
    load_dotenv()

    if isinstance(image_input, (bytes, bytearray)):
        image_bytes = bytes(image_input)
    elif isinstance(image_input, str):
        with open(image_input, "rb") as f:
            image_bytes = f.read()
    else:
        raise TypeError("image_input must be bytes, bytearray, or a file path string")

    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    json_schema = image_to_json(image_b64)
    print("Extracted JSON Schema:")
    print(json_schema)
    
    csv_output = json_to_csv(image_b64, json_schema)
    return csv_output

