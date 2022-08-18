import onnx
import onnxruntime as ort
import time
import cv2
import numpy as np
import random
from PIL import Image
import tensorflow as tf
import coremltools
import matplotlib.pyplot as plt
import argparse
import sys
import onnx
from onnx_tf.backend import prepare

def representative_dataset_gen(img):
    # Representative dataset generator for use with converter.representative_dataset, returns a generator of np arrays
    # im = np.transpose(img, [1, 2, 0])
    im=img
    im = np.expand_dims(im, axis=0).astype(np.float32)
    im /= 255
    yield [im]
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./best.onnx', help='weights path')
    parser.add_argument('--data', type=str, default='./example.png', help='data required for quantization')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--include-nms', action='store_true', help='include nms on the tflite')
    parser.add_argument('--int8', action='store_true', help='INT8 quantization')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
    print(opt)
    t = time.time()
    #load onnx model
    try:
        onnx_model = onnx.load("best.onnx")
    except Exception as e:
        print('ONNX load failure: %s' % e)
    #convert to tf model
    try:
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph("best_tf.pb")
    except Exception as e:
        print('TF export failure: %s' % e)
    #convert to tf lite
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model("best_tf.pb")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.target_spec.supported_types = [tf.float16]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # dataset = LoadImages(check_dataset(check_yaml(data))['train'], img_size=opt.img_size, auto=False)

        if opt.int8:
            converter.target_spec.supported_types = []
            image=cv2.imread(opt.data)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose((2, 0, 1))
            image = np.ascontiguousarray(image)
            im = image.astype(np.float32)
            im =im/255
            converter.representative_dataset = lambda: representative_dataset_gen(im)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.int8
            converter.inference_output_type = tf.int8  # or tf.int8
            converter.experimental_new_quantizer = True
        if opt.include_nms:  
            converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)
        tflite_model = converter.convert()
    
        open('best-tflite.tflite', "wb").write(tflite_model)
    except Exception as e:
        print('TFlite export failure: %s' % e)
        
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))