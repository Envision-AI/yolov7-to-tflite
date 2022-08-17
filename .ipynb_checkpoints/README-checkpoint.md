# Yolov7-tflite-conversion


This repo is for converting yolov7 onnx exported model into TFlite.

On the yolov7 repo export your model to onnx by using:


python3 export.py --weights best.pt --grid --end2end --simplify --topk-all 100 --conf-thres 0.35 --img-size 320 320 --max-wh 320



Afterwards use export.py on this repo to convert your model to TFlite

# Colab Walkthrough Tutorial

To make sure your model has converted properly follow the walkthrough tutorial and check after every conversion to see if there is a problem.

