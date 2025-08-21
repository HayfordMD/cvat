FROM python:3.8-slim

WORKDIR /opt/nuclio

# Install system dependencies
RUN apt-get update && apt-get -y install \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libglu1 libgl1 wget git gcc g++ libpython3-dev

# Install Python packages
RUN pip install --no-cache-dir \
    torch==1.13.0+cpu torchvision==0.14.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir \
    opencv-python pillow easyocr pyyaml numpy

# Copy function files
COPY nuclio/*.py ./
COPY nuclio/function-cpu.yaml ./function.yaml

# Create a simple HTTP server wrapper
RUN echo 'import json\n\
import base64\n\
from http.server import BaseHTTPRequestHandler, HTTPServer\n\
import main\n\
import io\n\
\n\
class MockContext:\n\
    def __init__(self):\n\
        self.logger = self\n\
        self.user_data = self\n\
        self.model = None\n\
    def info(self, msg):\n\
        print(msg)\n\
    def Response(self, body, headers, content_type, status_code):\n\
        return {"body": body, "status": status_code}\n\
\n\
class MockEvent:\n\
    def __init__(self, body):\n\
        self.body = body\n\
\n\
context = MockContext()\n\
main.init_context(context)\n\
\n\
class RequestHandler(BaseHTTPRequestHandler):\n\
    def do_POST(self):\n\
        content_length = int(self.headers["Content-Length"])\n\
        post_data = self.rfile.read(content_length)\n\
        data = json.loads(post_data)\n\
        \n\
        event = MockEvent(data)\n\
        response = main.handler(context, event)\n\
        \n\
        self.send_response(200)\n\
        self.send_header("Content-Type", "application/json")\n\
        self.end_headers()\n\
        self.wfile.write(response["body"].encode())\n\
    \n\
    def log_message(self, format, *args):\n\
        return\n\
\n\
if __name__ == "__main__":\n\
    server = HTTPServer(("0.0.0.0", 8080), RequestHandler)\n\
    print("Jersey OCR server running on port 8080")\n\
    server.serve_forever()\n\
' > server.py

EXPOSE 8080

CMD ["python", "server.py"]