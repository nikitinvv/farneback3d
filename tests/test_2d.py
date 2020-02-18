#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import farneback3d
import numpy as np

import dxchange
__author__ = "Stephan Seitz"
__copyright__ = "Stephan Seitz"
__license__ = "none"

TOLERANCE = 1e-2


def test_moving_cube():

    flow2d = True
    movement_vector = [0, 5, 1]    
    vol0 = np.zeros([8, 2048, 2048], np.float32)
    vol0[0, 10:15, 20:30] = 100.
    flow_ground_truth = np.stack([movement_vector[2] * np.ones(vol0.shape, np.float32), movement_vector[1] * np.ones(
        vol0.shape, np.float32), movement_vector[0] * np.ones(vol0.shape, np.float32)], 0)
    vol1 = farneback3d.warp_by_flow(vol0, flow_ground_truth)
    optflow = farneback3d.Farneback(
        levels=0, num_iterations=4, winsize=64, poly_n=5, quit_at_level=-1, use_gpu=True, fast_gpu_scaling=True)
    flow = optflow.calc_flow(vol0, vol1, flow2d)
    
    for axis in range(len(movement_vector)):
        print(np.max(flow[2-axis]) - movement_vector[axis])
        assert abs(np.max(flow[2-axis]) - movement_vector[axis]) < TOLERANCE

if __name__ == "__main__":
    test_moving_cube()
    # test_moving_cube_larger_distance()
    #test_default_values()
