from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw

# from yolov3_to_onnx import download_file
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import sys, os

# sys.path.insert(1, os.path.join(sys.path[0], ".."))
from common import *


def allocate_buffers(engine):

    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:

        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))
        # print(bindings)
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream
# @profile
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):

    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)

    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

    stream.synchronize()

    return [out.host for out in outputs]
#------------

TRT_LOGGER = trt.Logger()

# @profile
def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    '''
    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    '''
    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw

# @profile
def get_engine(onnx_file_path, FLAGS, engine_file_path=""):
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder,\
              builder.create_network() as network, \
              trt.OnnxParser(network, TRT_LOGGER) as parser:

            builder.max_workspace_size = (FLAGS.vram)* 1 << 30 # 1GB
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

            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")

            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if FLAGS.build:
        return build_engine()
    else:
        if os.path.exists(engine_file_path):
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, \
                 trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine()

# @profile
def main(FLAGS):
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""
    onnx_file_path = 'yolov3.onnx'
    engine_file_path = "yolov3.trt"
    input_image_path = 'debug_image/test1.jpg'
    input_resolution_yolov3_HW = (608, 608)
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    image_raw, image = preprocessor.process(input_image_path)

    shape_orig_WH = image_raw.size
    
    trt_outputs = []

    with get_engine(onnx_file_path, FLAGS, engine_file_path) as engine, \
        engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        
        # print('Running inference on image {}...'.format(input_image_path))
        max_batch_size = engine.max_batch_size
        image=np.tile(image,[36,1,1,1])
        inputs[0].host = image

        inf_batch = 36
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=inf_batch)

    output_shapes = [(max_batch_size, 255, 19, 19), (max_batch_size, 255, 38, 38), (max_batch_size, 255, 76, 76)]

    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]  # [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]

    postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    
                          "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), 
                                           (59, 119), (116, 90), (156, 198), (373, 326)],
                          "obj_threshold": 0.6, 
                          "nms_threshold": 0.5, 
                          "yolo_input_resolution": input_resolution_yolov3_HW}

    postprocessor = PostprocessYOLO(**postprocessor_args)

    feat_batch = [[trt_outputs[j][i] for j in range(len(trt_outputs))] for i in range(len(trt_outputs[0]))]

    for idx, layers  in enumerate(feat_batch):
        boxes, classes, scores = postprocessor.process(layers, (shape_orig_WH))

    # obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)
    # output_image_path = 'dog_bboxes.png'
    # obj_detected_img.save(output_image_path, 'PNG')

    # print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action="store_true", required=False, default=False,
                        help='Force to build model')
    parser.add_argument('--vram', type=int, required=False,
                        help='suppose using VRAM size.')
    parser.add_argument('--max_batch_size', type=int, required=False,
                        help='max batch size for Dynamic Batcher on TRTIS for TensorRT Runtime.')    
    parser.add_argument('-p', '--precision', type=str, choices=['NONE', 'fp16', 'int8'],
                        required=False, default='NONE',
                        help='dtype precision')

    FLAGS = parser.parse_args()

    main(FLAGS)