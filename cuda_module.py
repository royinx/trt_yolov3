from pycuda.compiler import SourceModule

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

__global__ void YoloResize(unsigned char* src_img, unsigned char* dst_img, 
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