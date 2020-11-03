import os 

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy as np
import cv2
from line_profiler import LineProfiler

from common import *
from data_processing import PostprocessYOLO
from onnx_to_tensorrt import draw_bboxes

profile = LineProfiler()

module = SourceModule("""

__device__ float lerp1d(int a, int b, float w)
{
    if(b>a){
        return a + w*(b-a);
    }
    else{
        return b + w*(a-b);
    }
}

__device__ float lerp2d(int f00, int f01, int f10, int f11,
                        float centroid_h, float centroid_w )
{
    centroid_w = (1 + lroundf(centroid_w) - centroid_w)/2;
    centroid_h = (1 + lroundf(centroid_h) - centroid_h)/2;
    
    float r0, r1, r;
    r0 = lerp1d(f00,f01,centroid_w);
    r1 = lerp1d(f10,f11,centroid_w);

    r = lerp1d(r0, r1, centroid_h); //+ 0.00001
    return r;
}

__global__ void Transpose(unsigned char *odata, const unsigned char *idata)
{
    int H = blockDim.x * gridDim.x; // # dst_height
    int W = blockDim.y * gridDim.y; // # dst_width 
    int h = blockDim.x * blockIdx.x + threadIdx.x;  // 32 * bkIdx[0:18] + tdIdx; [0,607]   # x / h-th row
    int w = blockDim.y * blockIdx.y + threadIdx.y;  // 32 * bkIdx[0:18] + tdIdx; [0,607]   # y / w-th col
    int C = 3; // # ChannelDim
    int c = blockIdx.z % 3 ; // [0,2] # ChannelIdx
    int n = blockIdx.z / 3 ; // [0 , Batch size-1], # BatchIdx

    long src_idx = n * (H * W * C) + 
                    h * (W * C) +
                    w * C +
                    c;

    long dst_idx = n * (C * H * W) +
                    c * (H * W)+
                    h * W+
                    w;

    odata[dst_idx] = idata[src_idx];
}

__global__ void Transpose_and_normalise(float *odata, const unsigned char *idata)
{
    int H = blockDim.x * gridDim.x; // # dst_height
    int W = blockDim.y * gridDim.y; // # dst_width 
    int h = blockDim.x * blockIdx.x + threadIdx.x;  // 32 * bkIdx[0:18] + tdIdx; [0,607]   # x / h-th row
    int w = blockDim.y * blockIdx.y + threadIdx.y;  // 32 * bkIdx[0:18] + tdIdx; [0,607]   # y / w-th col
    int C = 3; // # ChannelDim
    int c = blockIdx.z % 3 ; // [0,2] # ChannelIdx
    int n = blockIdx.z / 3 ; // [0 , Batch size-1], # BatchIdx

    long src_idx = n * (H * W * C) + 
                    h * (W * C) +
                    w * C +
                    c;

    long dst_idx = n * (C * H * W) +
                    c * (H * W)+
                    h * W+
                    w;

    odata[dst_idx] = idata[src_idx]/255.0;
}

__global__ void YoloResize(unsigned char* dst_img, unsigned char* src_img, 
                       int src_h, int src_w, 
                       int frame_h, int frame_w, 
                       float stride_h, float stride_w)
{
    int H = blockDim.x * gridDim.x; // # dst_height
    int W = blockDim.y * gridDim.y; // # dst_width 
    int h = blockDim.x * blockIdx.x + threadIdx.x;  // 32 * bkIdx[0:18] + tdIdx; [0,607]   # x / h-th row
    int w = blockDim.y * blockIdx.y + threadIdx.y;  // 32 * bkIdx[0:18] + tdIdx; [0,607]   # y / w-th col
    int C = 3; // # ChannelDim
    int c = blockIdx.z % 3 ; // [0,2] # ChannelIdx
    int n = blockIdx.z / 3 ; // [0 , Batch size-1], # BatchIdx
    
    int idx = n * (H * W * C) + 
              h * (W * C) +
              w * C +
              c;

    float centroid_h, centroid_w;  
    centroid_h = stride_h * (h + 0.5); // h w c -> x, y, z : 1080 , 1920 , 3
    centroid_w = stride_w * (w + 0.5); // 

    int f00,f01,f10,f11;

    int src_h_idx = lroundf(centroid_h)-1;
    int src_w_idx = lroundf(centroid_w)-1;
    if (src_h_idx<0){src_h_idx=0;}
    if (src_w_idx<0){src_w_idx=0;}

    f00 = n * frame_h * frame_w * C + 
          src_h_idx * frame_w * C + 
          src_w_idx * C +
          c;
    f01 = n * frame_h * frame_w * C +
          src_h_idx * frame_w * C +
          (src_w_idx+1) * C +
          c;
    f10 = n * frame_h * frame_w * C +
          (src_h_idx+1) * frame_w * C +
          src_w_idx * C +
          c;
    f11 = n * frame_h * frame_w * C + 
          (src_h_idx+1) * frame_w * C +
          (src_w_idx+1) * C +
          c;
          
    int rs = lroundf(lerp2d(src_img[f00], src_img[f01], src_img[f10], src_img[f11], 
                            centroid_h, centroid_w));

    dst_img[idx] = (unsigned char)rs;
}
    """)

# block = (32, 32, 1)   blockDim | threadIdx 
# grid = (19,19,3))     gridDim  | blockIdx

YoloResizeKer = module.get_function("YoloResize")
TransposeKer = module.get_function("Transpose")
TransNorKer = module.get_function("Transpose_and_normalise")

# post processor
postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    
                    "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), 
                                    (59, 119), (116, 90), (156, 198), (373, 326)],
                    "obj_threshold": 0.6,
                    "nms_threshold": 0.5,
                    "yolo_input_resolution": (608, 608)}
postprocessor = PostprocessYOLO(**postprocessor_args)


# TRT init
def get_engine(engine_file_path):
    if os.path.exists(engine_file_path):
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, \
            trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print("Cannot find .trt engine file.")
        exit(0)

TRT_LOGGER = trt.Logger()
max_batch_size = 40
engine = get_engine("yolov3.trt")
context = engine.create_execution_context()
max_batch_size = engine.max_batch_size
bindings = []




# run preprocess and inference
@profile
def gpu_resize(input_img: np.ndarray , stream: cuda.Stream):
    """
    Resize the batch image to (608,608) 
    and Convert NHWC to NCHW
    pass the gpu array to normalize the pixel ( divide by 255)

    Application oriented

    input_img : batch input, format: NHWC , recommend RGB. *same as the NN input format 
                input must be 3 channel, kernel set ChannelDim as 3.
    out : batch resized array, format: NCHW , same as intput channel
    """
    # ========= Init Params =========
    # stream = cuda.Stream()

    # convert to array
    batch, src_h, src_w, channel = input_img.shape
    dst_h, dst_w = 608, 608
    frame_h, frame_w = 1080*2, 1920*2
    assert (src_h <= frame_h) & (src_w <= frame_w)
    # Mem Allocation
    # input memory
    
    inp = {"host":cuda.pagelocked_zeros(shape=(batch,frame_h,frame_w,channel),
                                        dtype=np.uint8,
                                        mem_flags=cuda.host_alloc_flags.DEVICEMAP)}
    inp["host"][:,:src_h,:src_w,:] = input_img
    inp["device"] = cuda.mem_alloc(inp["host"].nbytes)
    cuda.memcpy_htod_async(inp["device"], inp["host"],stream)


    # output data
    out = {"host":cuda.pagelocked_zeros(shape=(batch,dst_h,dst_w,channel), 
                                    dtype=np.uint8,
                                    mem_flags=cuda.host_alloc_flags.DEVICEMAP)}
    out["device"] = cuda.mem_alloc(out["host"].nbytes)
    cuda.memcpy_htod_async(out["device"], out["host"],stream)


    #Transpose (and Normalize)
    trans = {"host":cuda.pagelocked_zeros(shape=(batch,channel,dst_h,dst_w), 
                                            dtype=np.float32,
                                            mem_flags=cuda.host_alloc_flags.DEVICEMAP)}  # N C H W
    trans["device"] = cuda.mem_alloc(trans["host"].nbytes)
    cuda.memcpy_htod_async(trans["device"], trans["host"],stream)
    
    
    # YoloOutput
    # yolo_out = {"host":cuda.pagelocked_zeros(shape=(batch,channel,dst_h,dst_w), 
    #                                         dtype=np.float32,
    #                                         mem_flags=cuda.host_alloc_flags.DEVICEMAP)}  # N C H W
    # yolo_out["device"] = cuda.mem_alloc(yolo_out["host"].nbytes)
    # cuda.memcpy_htod_async(yolo_out["device"], yolo_out["host"],stream)


    outputs = []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        if engine.binding_is_input(binding):
            bindings.append(int(trans["device"]))
        else:
            bindings.append(int(device_mem))
            outputs.append(HostDeviceMem(host_mem, device_mem))

    
    # init resize , store kernel in cache
    print(batch)
    YoloResizeKer(out["device"], inp["device"], 
               np.int32(src_h), np.int32(src_w),
               np.int32(frame_h), np.int32(frame_w),
               np.float32(src_h/dst_h), np.float32(src_w/dst_w),
               block=(32, 32, 1),
               grid=(19,19,3*batch),
               stream=stream)

    # ========= Testing =========

    for _ in range(25):
        YoloResizeKer(out["device"], inp["device"],
                        np.int32(src_h), np.int32(src_w),
                        np.int32(frame_h), np.int32(frame_w),
                        np.float32(src_h/dst_h), np.float32(src_w/dst_w),
                        block=(32, 32, 1),
                        grid=(19,19,3*batch),
                        stream=stream)

    # ========= Copy out result =========

    TransNorKer(trans["device"],out["device"],
                block=(32, 32, 1),
                grid=(19,19,3*batch))


    context.execute_async(batch_size=max_batch_size, 
                            bindings=bindings, 
                            stream_handle=stream.handle)



    cuda.memcpy_dtoh_async(trans["host"], trans["device"],stream)


    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]


    stream.synchronize()

    trt_outputs = [out.host for out in outputs]
    for i in trt_outputs:
        print(i.shape)
    output_shapes = [(max_batch_size, 255, 19, 19), 
                    (max_batch_size, 255, 38, 38), 
                    (max_batch_size, 255, 76, 76)]

    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]  # [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]

    feat_batch = [[trt_outputs[j][i] for j in range(len(trt_outputs))] for i in range(len(trt_outputs[0]))]
    post = [[postprocessor.process(feat,input_img.shape)]for feat in feat_batch] # out:[[bbox,score,categories,confidences],...]
    post = post[:len(input_img)]

    im = input_img[0]
    for feat in post[0][0]:
        x1,y1,x2,y2,_,_ = feat.astype(np.int16)
        im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # out: array with x,y coordinate in axis=1  / (N_obj, 4) = [(x1,y1,x2,y2,class,cls_score), ...]
    # boxes, classes, scores = postprocessor.process(feats, (shape_orig_WH))
    cv2.imwrite("output.jpg",im)
    exit()

    return trans["host"]

if __name__ == "__main__":
    # init
    stream = cuda.Stream()

    grid = 19
    block = 32
    batch = 2

    # img = cv2.resize(cv2.imread("trump.jpg"),(1920,1080))
    # img = cv2.imread("trump.jpg")
    # img = np.tile(img,[batch,1,1,1])

    # img = np.zeros(shape=(3,1080,1920,3),dtype = np.uint8)
    # img[0,:48,:64,:] = cv2.resize(cv2.imread("trump.jpg"),(64,48))
    # img[1,:480,:640,:] = cv2.resize(cv2.imread("trump.jpg"),(640,480))
    # img[2,:1080,:1920,:] = cv2.resize(cv2.imread("trump.jpg"),(1920,1080))

    batch = 40
    # img_batch_0 = np.tile(cv2.resize(cv2.imread("trump.jpg"),(64,48)),[batch,1,1,1])
    # img_batch_1 = np.tile(cv2.resize(cv2.imread("trump.jpg"),(320,240)),[batch,1,1,1])
    img_batch_2 = np.tile(cv2.imread("debug_image/two_face.jpg"),[batch,1,1,1])
    # pix_0 = gpu_resize(img_batch_0)
    # pix_1 = gpu_resize(img_batch_1)
    pix_2 = gpu_resize(img_batch_2,stream = stream)
    # if normalize or transpose:
    if True:        
        # pix_0 = np.transpose(pix_0,[0,2,3,1])
        # pix_1 = np.transpose(pix_1,[0,2,3,1])
        pix_2 = np.transpose(pix_2,[0,2,3,1])
    # cv2.imwrite("trans0.jpg", pix_0[0])
    # cv2.imwrite("trans1.jpg", pix_1[0])
    cv2.imwrite("trans2.jpg", pix_2[0])

    profile.print_stats()
    # print(pix.shape)
    # cv2.imwrite("pycuda_outpuut.jpg", pix[0])