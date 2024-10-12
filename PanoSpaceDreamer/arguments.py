###
# Copyright (C) 2023, Computer Vision Lab, Seoul National University, https://cv.snu.ac.kr
# For permission requests, please contact robot0321@snu.ac.kr, esw0116@snu.ac.kr, namhj28@gmail.com, jarin.lee@gmail.com.
# All rights reserved.
###
import numpy as np


class GSParams: 
    def __init__(self):
        self.sh_degree = 3
        self.images = "images"
        self.resolution = -1
        self.white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.use_depth = False

        self.iterations = 2990#3_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 2990#3_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

class CameraParams:
    def __init__(self, H: int = 768, W: int = 768, fov_deg: float = 49.9):
        self.H = H
        self.W = W
        fov_rad = np.radians(fov_deg)
        self.focal = (self.W / (2 * np.tan(fov_rad / 2)), self.H / (2 * np.tan(fov_rad / 2)))
        self.fov = (fov_rad, fov_rad)
        self.K = np.array([
            [self.focal[0], 0., self.W / 2],
            [0., self.focal[1], self.H / 2],
            [0., 0., 1.],
        ]).astype(np.float32)