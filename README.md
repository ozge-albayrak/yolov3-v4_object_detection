# yolov3-v4_object_detection

## Real Time Object Detection using YOLOv3 & v4 with OpenCV and Training Custom Dataset on Google Colab 
This project implements an image and video capture (real-time with OpenCV), using WebCam or Intel RealSense LiDAR Camera, object detection classifier first using pre-trained yolov3 and yolov4 models and then our custom model to detect custom objects. 

Additionally, an effective and straightforward approach for training your custom dataset for object detection on Google Colab with yolov3 & v4 using the Darknet library has been implemented. 


## Train an object detector using YOLO on Google Drive 

1. Git clone the repo. 
```
git clone https://github.com/AlexeyAB/darknet

```

## How to prepare the custom .cfg file for training YOLO
You need to make the following changes in your config file (e.g. yolov4-custom.cfg):

Under # Training 
- change line batch to batch=64
- change line submissions to submissions=16 
- set network size width=416, height=416 or any value multiple of 32

- change line max_batches to (classes*2000) but not less than the number of training images and not less than 6000), exp. max_batches=6000 if you train for 3 classes
- change the line steps to 80% and 90% of max_batches, exp. steps=4800,5400

- change filters=255 to filters=(classes+5)*3 in the last "3" [convolutional] layers BEFORE each [yolo] layer. 
- change line classes=80 to your number of objects in each of "3" [yolo] layers.  

So, if classes=1 then, filters=18. If classes=2 then, filters=21 and so on.




