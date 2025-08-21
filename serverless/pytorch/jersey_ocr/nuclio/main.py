import json
import base64
from PIL import Image
import io
import yaml
import os

# Try to import the full handler, fall back to simple if needed
try:
    from model_handler import ModelHandler
except ImportError:
    from model_handler_simple import ModelHandler

def init_context(context):
    context.logger.info("Init context... 0%")
    
    # Read labels from function.yaml
    with open("/opt/nuclio/function.yaml", 'rb') as function_file:
        functionconfig = yaml.safe_load(function_file)
    
    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}
    
    # Initialize the model handler
    model = ModelHandler(labels)
    context.user_data.model = model
    
    context.logger.info("Init context... 100%")

def handler(context, event):
    context.logger.info("Run Jersey OCR model")
    data = event.body
    
    # Decode the input image
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    
    # Get optional parameters
    confidence_threshold = float(data.get("confidence", 0.5))
    
    # Run inference
    results = context.user_data.model.infer(image, confidence_threshold)
    
    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type='application/json',
        status_code=200
    )