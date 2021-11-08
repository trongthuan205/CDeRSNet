import warnings
warnings.filterwarnings("ignore")
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os.path
from mmdet.models.detectors import BaseDetector
#BaseDetector.show_result instead of show_result
#import os.path as osp
import pickle
import shutil
import tempfile
import time
from PIL import Image


import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from numpy import asarray
from mmcv.utils import is_str
from tqdm import tqdm

#from mmdet.core import encode_mask_results, tensor2imgs
config_fname = 'work_dirs/htc_x101_dcn/htc_x101_dcn.py'
checkpoint_fname = 'work_dirs/htc_x101_dcn/best_bbox_mAP.pth'
model = init_detector(config_fname, checkpoint_fname)

import os
imgFolderPath = "UIT-DODV/images/test/"

outputFolderPath = "dcn/"
print(outputFolderPath)
if not os.path.exists(outputFolderPath):
    os.makedirs(outputFolderPath)
    
for file in tqdm(os.listdir(imgFolderPath)):
  if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".JPG"):
    img = os.path.join(imgFolderPath, file)
    result = inference_detector(model, img)
    model.show_result(img, result, out_file= outputFolderPath + file)