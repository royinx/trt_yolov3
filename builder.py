from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import sys, os

from common import *
from turbojpeg import TurboJPEG
from line_profiler import LineProfiler
import time 
# import cv2

jpeg = TurboJPEG()

profile = LineProfiler()

def build_engine(FLAGS):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    onnx_file_path = FLAGS.onnx
    TRT_LOGGER = trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder,\
            builder.create_network() as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        builder.max_workspace_size = FLAGS.vram* 1 << 30 # 1GB
        builder.max_batch_size = FLAGS.max_batch_size

        if FLAGS.precision == 'fp16':
            # set to fp16 
            print('force to fp16')
            builder.fp16_mode = True
            builder.strict_type_constraints = True
        elif FLAGS.precision == 'int8':
            # set to int8

            pass
            # builder.int8_mode = True

            '''
            NUM_IMAGES_PER_BATCH = 5 
            batch = ImageBatchStream(NUM_IMAGES_PER_BATCH, calibration_files)
            Int8_calibration = EntropyCalibrator(['input_node_name'],batchstream)
            trt_builder.int8_calibrator = Int8_calibrator
            '''
        else:
            pass

        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
            exit(0)

        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        print('Completed parsing of ONNX file')

        # setting output layer
        # if FLAGS.onnx=="yolov3.onnx":
        #     print(network.num_layers - 1)
        #     output_1 = network.get_layer(82)
        #     output_1 = network.get_layer(-1)
        #     print(output_1)
        #     # output_2 = network.get_layer(94)
        #     # output_3 = network.get_layer(106)
        #     # network.mark_output(output_1.get_output(0),output_2.get_output(0),output_3.get_output(0))
        #     exit()
        # else:
        #     last_layer = network.get_layer(network.num_layers - 1)
        #     network.mark_output(last_layer.get_output(0))


        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")

        engine_file_path = f'{onnx_file_path.split(".onnx")[0]}_batch_{FLAGS.max_batch_size}.trt'
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine

def main(FLAGS):
    if FLAGS.build:
        build_engine(FLAGS)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action="store_true", required=False, default=False,
                        help='Build model')
    parser.add_argument('--vram', type=int, required=False,
                        help='(Build mode) int - Suppose using VRAM size, recommand half of GPU memory.')
    parser.add_argument('--max_batch_size', type=int, required=False,
                        help='(Build mode) Max batch size for memalloc on GPU.')    
    parser.add_argument('-p', '--precision', type=str, choices=['fp32', 'fp16', 'int8'],
                        required=False, default='fp16',
                        help='(Build mode) dtype precision')
    parser.add_argument('--onnx', type=str,
                        required=False, default='yolov3.onnx',
                        help='(Build mode) ONNX model location')

    FLAGS = parser.parse_args()

    main(FLAGS)

    # docker build -t yolotrt . 

    # Unit test
    # docker run -it --rm --runtime=nvidia yolotrt python3 refactor.py 

    # Build engine
    # docker run -it --rm --runtime=nvidia -v ${PWD}:/workshop -w /workshop   yolotrt python3 builder.py --build --vram 2 --max_batch_size 5
