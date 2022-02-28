"""
This is the main module for linking different components of the CNN-based model
proposed for the task of image defocus deblurring based on dual-pixel data. 

Copyright (c) 2020-present, Abdullah Abuolaim
This source code is licensed under the license found in the LICENSE file in
the root directory of this source tree.

This code imports the modules and starts the implementation based on the
configurations in config.py module.

Note: this code is the implementation of the "Defocus Deblurring Using Dual-
Pixel Data" paper accepted to ECCV 2020. Link to GitHub repository:
https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel

Email: abuolaim@eecs.yorku.ca
"""

from model import *
from config import *
from data import *
import os

check_dir(path_write)
data_random_shuffling('test')
model = load_model(path_save_model, compile=False)
# fix input layer size
model.layers.pop(0)
input_size = (img_h, img_w, nb_ch_all)
input_test = Input(input_size)
output_test=model(input_test)
model = Model(input = input_test, output = output_test)

img_mini_b = 1

test_imgaes, gt_images = test_generator(total_nb_test)
predictions = model.predict(test_imgaes,img_mini_b,verbose=1)
                        
save_eval_predictions(path_write,test_imgaes,predictions,gt_images)

np.save(path_write+'mse_arr',np.asarray(mse_list))
np.save(path_write+'psnr_arr',np.asarray(psnr_list))
np.save(path_write+'ssim_arr',np.asarray(ssim_list))
np.save(path_write+'mae_arr',np.asarray(mae_list))
np.save(path_write+'final_eval_arr',[np.mean(np.asarray(mse_list)),
                                        np.mean(np.asarray(psnr_list)),
                                        np.mean(np.asarray(ssim_list)),
                                        np.mean(np.asarray(mae_list))])