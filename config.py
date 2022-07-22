"""
This is the configuration module has all the gobal variables and basic
libraries to be shared with other modules in the same project.

Copyright (c) 2020-present, Abdullah Abuolaim
This source code is licensed under the license found in the LICENSE file in
the root directory of this source tree.

Note: this code is the implementation of the "Defocus Deblurring Using Dual-
Pixel Data" paper accepted to ECCV 2020. Link to GitHub repository:
https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel

Email: abuolaim@eecs.yorku.ca
"""

import numpy as np
import os
import math
import cv2
import random
from skimage import measure
from sklearn.metrics import mean_absolute_error
from utils import *

# results and model name
res_model_name='l5_s512_f0.7_d0.4'
op_phase='test' 

# input image size
img_w = 1680
img_h = 1120
#########################################################################
# READ & WRITE DATA PATHS									            #
#########################################################################
# run on server or local machine
server=False

sub_folder=['source/','target/']

if op_phase=='test':
    dataset_name='dpd'
    # resize flag to resize input and output images
    resize_flag=False         
else:
    raise NotImplementedError

# path to save model
if server:
    path_save_model='/local/ssd/abuolaim/defocus_deblurring_dp_'+res_model_name+'.hdf5'
else:
    path_save_model='./ModelCheckpoints/defocus_deblurring_dp_'+res_model_name+'.hdf5'
    

path_read_test = './DPD/test_c/source' 
path_read_test_gt = './DPD/test_c/target'

# path to write results
path_write='./results/res_'+res_model_name+'_dd_dp'+dataset_name+'/'

#########################################################################
# NUMBER OF IMAGES IN THE TRAINING, VALIDATION, AND TESTING SETS	    #
#########################################################################
total_nb_test = len(make_dataset(path_read_test))

#########################################################################
# MODEL PARAMETERS & TRAINING SETTINGS									#
#########################################################################
# input patch size
patch_w=512
patch_h=512

# mean value pre-claculated
src_mean=0
trg_mean=0

# number of input channels
nb_ch_all= 3
# number of output channels
nb_ch=3  # change conv9 in the model and the folowing variable

# color flag:"1" for 3-channel 8-bit image or "0" for 1-channel 8-bit grayscale
# or "-1" to read image as it including bit depth
color_flag = 1#-1

bit_depth = 8

norm_val=(2**bit_depth)-1

train_set, val_set, test_set = [], [], []

size_set, portrait_orientation_set = [], []

mse_list, psnr_list, ssim_list, mae_list = [], [], [], []
