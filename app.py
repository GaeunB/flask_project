#Flask Server

from urllib import request
import urllib.request
import cv2
import numpy as np
import base64
from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import requests
import time
from collections import Counter
from threading import Thread


app = Flask(__name__)


# React server & CORS
CORS(app, resources={r"*": {"origins": ["http://192.168.43.192:3005"]}})

# # Arduino webserver URL
# arduino_url = 'http://192.168.43.101/'

# Importing a YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='your_path/yolov5_cupramen_best.pt') # 경로 설정 필요
# Set the detection threshold
model.conf = 0.5

# Variables for counting objects
last_time = 0
print_interval = 3  # output interval
latest_detected_object_names = None  # Variable to store the name of the last detected class
latest_detected_time = 0    # Variable to store the last detected time
first_detection_done = False   # Add a variable to indicate whether the first object detection is complete
count = 0
defective_count = 0
count_from_arduino = 0
timestamp_from_arduino = None
defective_count_from_arduino = 0
pre_count = 0
serial_counter = 1  # Generate a serial number for an item (set the starting number for the serial number)

def fetch_and_process_image():  # Fetching and processing images from the Arduino server
    img_resp = urllib.request.urlopen(arduino_url + 'cam-hi.jpg')
    img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(img_np, -1)
    new_width = 640
    new_height = 480
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    blurred_image = cv2.GaussianBlur(img, (5, 5), 0)
    sharpened_image = cv2.addWeighted(img, 1.5, blurred_image, -0.5, 0)

    return sharpened_image

# Get the object count value and timestamp value from the Arduino ESP32 board
@app.route('/count', methods=['POST'])
def receive_data_from_arduino():
    global count_from_arduino, timestamp_from_arduino, defective_count, defective_count_from_arduino

    json_data = request.get_json()

    count_from_arduino = json_data['count']
    timestamp_from_arduino = json_data['times']

    print("count_from_arduino: ", count_from_arduino)
    print("pre_count:", pre_count, "count_from_arduino:", count_from_arduino)

    return jsonify({"defective_count": defective_count}), 200


# Send an image to the React frontend server
@app.route('/get-live-transmission', methods=['GET'])
def get_live_transmission():
    sharpened_image = fetch_and_process_image()
    results = model(sharpened_image, size=416)
    encoded_img = process_and_encode_image(results)

    return jsonify({'image': encoded_img})

if __name__ == '__main__':
    t = Thread(target=continuous_object_detection_and_processing)
    t.start()  # Run the function in a separate thread
    app.run(host='0.0.0.0', debug=False, port=5000, threaded=True)