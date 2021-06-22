import pyrealsense2 as rs
from cv2 import cv2
import numpy as np
import sys, time, math

# Load yolov4
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

print(classes)

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Configure camera streams from Intel Camera
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

while True:
    # Wait for a coherent pair of frame: color only
    frame = pipeline.wait_for_frames()
    color_frame = frame.get_color_frame()
    if not color_frame:
        continue

    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    height, width, channels = color_image.shape
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Detecting objects
    blob = cv2.dnn.blobFromImage(
        color_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(color_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(color_image, "{} [{:.1f}%]".format(label, float(confidence) * 100), (x, y - 5), font, 2, color, 3)

    cv2.imshow("Frame", color_image)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
