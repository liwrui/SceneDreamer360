###
# Copyright (C) 2023, Computer Vision Lab, Seoul National University, https://cv.snu.ac.kr
# For permission requests, please contact robot0321@snu.ac.kr, esw0116@snu.ac.kr, namhj28@gmail.com, jarin.lee@gmail.com.
# All rights reserved.
###
import os
import random

from PanoSpaceDreamer.arguments import GSParams
from PanoSpaceDreamer.utils.system import searchForMaxIteration
from PanoSpaceDreamer.scene.dataset_readers import readDataInfo
from PanoSpaceDreamer.scene.gaussian_model import GaussianModel


class Scene:
    gaussians: GaussianModel

    def __init__(self, traindata, gaussians: GaussianModel, opt: GSParams):
        self.traindata = traindata
        self.gaussians = gaussians
        
        info = readDataInfo(traindata, opt.white_background)
        random.shuffle(info.train_cameras)  # Multi-res consistent random shuffling
        self.cameras_extent = info.nerf_normalization["radius"]

        print("Loading Training Cameras")
        self.train_cameras = info.train_cameras        
        print("Loading Preset Cameras")
        self.preset_cameras = {}
        print(f'info.preset_cameras.keys:{info.preset_cameras.keys()}')
        for campath in info.preset_cameras.keys():
            self.preset_cameras[campath] = info.preset_cameras[campath]

        self.gaussians.create_from_pcd(info.point_cloud, self.cameras_extent)
        self.gaussians.training_setup(opt)

    def getTrainCameras(self):
        return self.train_cameras
    
    def getPresetCameras(self, preset):
        assert preset in self.preset_cameras
        return self.preset_cameras[preset]