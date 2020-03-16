#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import farneback3d
import numpy as np
import dxchange

def test_moving_cube():

    movement_vector = [0.3, 0.1, 1]

    vol0 = np.zeros([64, 64, 64], np.float32)
    vol0[10:15, 20:25, 30:35] = 100.
    # flow_ground_truth = 4* np.ones([*vol0.shape,3], np.float32)
    flow_ground_truth = np.stack([movement_vector[2] * np.ones(vol0.shape, np.float32), movement_vector[1] * np.ones(
        vol0.shape, np.float32), movement_vector[0] * np.ones(vol0.shape, np.float32)], 0)

    vol1 = farneback3d.warp_by_flow(vol0, flow_ground_truth)

    optflow = farneback3d.Farneback(
        levels=2, num_iterations=5, poly_n=5, quit_at_level=-1, use_gpu=True, fast_gpu_scaling=True)
    flow = optflow.calc_flow(vol0, vol1)
    print(np.linalg.norm(flow))
    for axis in range(len(movement_vector)):
        print(np.max(flow[2-axis]) - movement_vector[axis])
        assert abs(np.max(flow[2-axis]) - movement_vector[axis]) < TOLERANCE

def test_real():

    a = np.ones([20] * 3)
    b = np.ones([20] * 3)

    optflow = farneback3d.Farneback()

    rtn = optflow.calc_flow(a, b)
    assert rtn is not None


if __name__ == "__main__":
#    test_moving_cube()
    optflow = farneback3d.Farneback(
        levels=5, num_iterations=5, poly_n=5, quit_at_level=-1, use_gpu=True, fast_gpu_scaling=True)

    vol0 = dxchange.read_tiff_stack('/home/beams0/VNIKITIN/farneback3d/tests/vols/registered time points_t001_z001.tif',ind = np.arange(10+16,522-16))
    vol1 = dxchange.read_tiff_stack('/home/beams0/VNIKITIN/farneback3d/tests/vols/registered time points_t002_z001.tif',ind = np.arange(10+16,522-16))
    vol2 = dxchange.read_tiff_stack('/home/beams0/VNIKITIN/farneback3d/tests/vols/registered time points_t003_z001.tif',ind = np.arange(10+16,522-16))


    vol0=vol0[::2,::2,::2].astype('float32')
    vol1=vol1[::2,::2,::2].astype('float32')
    vol2=vol2[::2,::2,::2].astype('float32')

    print(vol0.shape)
    print(vol1.shape)
    print(vol2.shape)


    print(np.linalg.norm(vol0))
    print(np.linalg.norm(vol1))
    print(np.linalg.norm(vol2))


    print(np.min(vol0))
    print(np.min(vol1))
    print(np.min(vol2))

    print(np.max(vol0))
    print(np.max(vol1))
    print(np.max(vol2))



    vol0=(vol0-np.min(vol0))/(np.max(vol0)-np.min(vol0))*255
    vol1=(vol1-np.min(vol1))/(np.max(vol1)-np.min(vol1))*255
    vol2=(vol2-np.min(vol2))/(np.max(vol2)-np.min(vol2))*255
    dxchange.write_tiff_stack(vol0,'/home/beams0/VNIKITIN/farneback3d/tests/vol0/rec___00000.tiff',overwrite=True)
    dxchange.write_tiff_stack(vol1,'/home/beams0/VNIKITIN/farneback3d/tests/vol1/rec___00000.tiff',overwrite=True)
    dxchange.write_tiff_stack(vol2,'/home/beams0/VNIKITIN/farneback3d/tests/vol2/rec___00000.tiff',overwrite=True)
    exit()

    flow = optflow.calc_flow(vol1, vol2)
    print(np.linalg.norm(flow))
    vol11 = farneback3d.warp_by_flow(vol1, flow)
    dxchange.write_tiff_stack(vol11,'/home/beams0/VNIKITIN/farneback3d/tests/res/rec___00000.tiff',overwrite=True)

    flow = optflow.calc_flow(vol0, vol2)
    print(np.linalg.norm(flow))
    vol02 = farneback3d.warp_by_flow(vol0, flow)
    dxchange.write_tiff_stack(vol02,'/home/beams0/VNIKITIN/farneback3d/tests/res/rec___00000.tiff',overwrite=True)
