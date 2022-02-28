"""
This module has all the functions used for the data manipulation, data
generation, and learning rate scheduler.

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

from config import *
from metrics import *
from utils import *

def check_dir(_path):
    if not os.path.exists(_path):
        try:
            os.makedirs(_path)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                

def data_random_shuffling(temp_type):
    global train_set, val_set, test_set, path_read_train, path_read_val_test
    if temp_type == 'test':
        path_read_input = path_read_test
        path_read_gt = path_read_test_gt

    images_C_src = make_dataset(path_read_input)

    images_C_src.sort()
    images_C_trg = make_dataset(path_read_gt)
    images_C_trg.sort()
    
    images_L_src = images_C_trg #(fake here)
    images_R_src = images_C_trg  # (fake here)


    
    len_imgs_list=len(images_C_src)
    
    # generate random shuffle index list for all list
    tempInd=np.arange(len_imgs_list)
    random.shuffle(tempInd)
    
    images_C_src=np.asarray(images_C_src)[tempInd]
    images_C_trg=np.asarray(images_C_trg)[tempInd]


    for i in range(len_imgs_list):
        if temp_type =='test':
            test_set.append([images_C_src[i],images_L_src[i],images_R_src[i],
                             images_C_trg[i]])
        else:
            raise NotImplementedError


def test_generator(num_image):
    in_img_tst = np.zeros((num_image, img_h, img_w, nb_ch_all))
    out_img_gt = np.zeros((num_image, img_h, img_w, nb_ch))

    for i in range(num_image):
        print('Read image: ',i,num_image)
        print('image path',test_set[i][0],test_set[i][3])
        in_img_tst[i, :] = cv2.imread(test_set[i][0],color_flag)/norm_val

        out_img_gt[i, :] = cv2.imread(test_set[i][3],color_flag)/norm_val
    
    return in_img_tst, out_img_gt
      
        
def save_eval_predictions(path_to_save,test_imgaes,predictions,gt_images):
    global mse_list, psnr_list, ssim_list, test_set
    for i in range(len(test_imgaes)):
        mse, psnr, ssim = MSE_PSNR_SSIM((gt_images[i]).astype(np.float64), (predictions[i]).astype(np.float64))
        mae = MAE((gt_images[i]).astype(np.float64), (predictions[i]).astype(np.float64))
        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        mae_list.append(mae)

        temp_in_img=cv2.imread(test_set[i][0],color_flag)
        if bit_depth == 8:
            temp_out_img=((predictions[i]*norm_val)+src_mean).astype(np.uint8)
            temp_gt_img=((gt_images[i]*norm_val)+src_mean).astype(np.uint8)

        img_name=((test_set[i][0]).split('/')[-1]).split('.')[0]
        cv2.imwrite(path_to_save+str(img_name)+'_p.png',temp_out_img)
        print('Write image: ',i,len(test_imgaes))
        
    print('image_quality_psnr_ssim_mae_mse',np.mean(psnr_list),np.mean(ssim_list),np.mean(mae_list),np.mean(mse_list))