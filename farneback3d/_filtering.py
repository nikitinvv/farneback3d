import os
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
from farneback3d._utils import divup

with open(os.path.join(os.path.dirname(__file__), 'filtering.cu')) as f:
    read_data = f.read()
f.closed

mod = SourceModule(read_data)


def _call_smooth_cuda_gauss(gpuimg, gpuresult, sigma, kernelsize,  filter_mask=None, flow2d=False):

    filter_nonzeros_only = filter_mask is not None
    if flow2d:   
        block = (32, 32, 1)
        grid = (int(divup(gpuimg.shape[2], block[0])),
                int(divup(gpuimg.shape[1], block[1])), 1)

        assert not filter_nonzeros_only, "not implemented yet for 2d"
        assert filter_mask is None
        #convolve2d = mod.get_function("convolve2d_gauss")
        #convolve2d(gpuimg,
         #          gpuresult,
                   #np.int32(gpuimg.shape[2]),
                   #np.int32(gpuimg.shape[1]),
                   #np.int32(gpuimg.shape[0]),
                   #np.int32(kernelsize),
                   #np.int32(kernelsize),
                   #np.float32(sigma * sigma),
                   #np.float32(sigma * sigma),
                   #grid=grid, block=block)   
        
        [x,y] = np.meshgrid(np.arange(gpuimg.shape[1])-gpuimg.shape[1]//2,np.arange(gpuimg.shape[2])-gpuimg.shape[2]//2)
        gauss = np.exp(-0.5/(sigma**2)*((x)**2+(y)**2)).astype('float32')                 
        crop = gauss*0 
        crop[gauss.shape[0]//2-kernelsize//2:gauss.shape[0]//2+kernelsize//2,gauss.shape[1]//2-kernelsize//2:gauss.shape[1]//2+kernelsize//2]=1
        gauss*=crop
        gauss/=np.sum(gauss)
        fgauss = np.tile(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(gauss))),[gpuimg.shape[0],1,1])
        cpuimg = gpuimg.get()
        fcpuimg = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(cpuimg)))
        res = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fcpuimg*fgauss)))        
        gpuresult.set(res.real)        
    else:
        block = (16, 16, 4)
        grid = (int(divup(gpuimg.shape[2], block[0])),
                int(divup(gpuimg.shape[1], block[1])), int(divup(gpuimg.shape[0], block[2])))

        if filter_mask is None:
            convolve3d = mod.get_function("convolve3d_gauss")
            convolve3d(gpuimg,
                       gpuresult,
                       np.int32(gpuimg.shape[2]),
                       np.int32(gpuimg.shape[1]),
                       np.int32(gpuimg.shape[0]),
                       np.int32(kernelsize),
                       np.int32(kernelsize),
                       np.int32(kernelsize),
                       np.float32(sigma * sigma),
                       np.float32(sigma * sigma),
                       np.float32(sigma * sigma),
                       np.int32(filter_nonzeros_only),
                       grid=grid, block=block)
        else:
            convolve3d = mod.get_function("convolve3d_gauss_with_mask")
            convolve3d(gpuimg,
                       gpuresult,
                       filter_mask,
                       np.int32(gpuimg.shape[2]),
                       np.int32(gpuimg.shape[1]),
                       np.int32(gpuimg.shape[0]),
                       np.int32(kernelsize),
                       np.int32(kernelsize),
                       np.int32(kernelsize),
                       np.float32(sigma * sigma),
                       np.float32(sigma * sigma),
                       np.float32(sigma * sigma),
                       np.int32(filter_nonzeros_only),
                       grid=grid, block=block)    

    return gpuresult


def smooth_cuda_gauss(img, sigma, kernelsize, rtn_gpu=None, filter_mask=None, flow2d=False):
    if not rtn_gpu:
        rtn_gpu = gpuarray.GPUArray(img.shape, np.float32)

    if isinstance(img, np.ndarray):

        img = np.ascontiguousarray(img, np.float32)

        img_gpu = gpuarray.to_gpu(img)
        _call_smooth_cuda_gauss(img_gpu, rtn_gpu, sigma,
                                kernelsize, filter_mask,flow2d)

        return rtn_gpu.get()
    elif isinstance(img, gpuarray.GPUArray):
        # TODO inplace algorithm
        _call_smooth_cuda_gauss(img, rtn_gpu,
                                sigma,  kernelsize, filter_mask, flow2d)

    return rtn_gpu
