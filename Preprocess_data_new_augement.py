# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:38:20 2019

@author: E75849
"""



import numpy as np
import os
from tensorflow import keras
from PIL import Image
import glob
import cv2
from matplotlib import pyplot as plt

import random
import argparse 

dimen = 400

#dir_path = 'Z:/Ravi K/ODIR_2019/Data/Classes/'
#save_path = 'Z:/Ravi K/ODIR_2019/Data/Resize/'
#sub_dir_list = os.listdir( dir_path )

dir_path = 'D:/ODIR_2019 Challenge/Data/ODIR_Data/dd/'
save_path = 'D:/ODIR_2019 Challenge/Data/ODIR_Data/dd/Resize/'
sub_dir_list = os.listdir( dir_path )



dimen= 512

#from glob import glob
#path_list = glob('D:/ODIR_2019 Challenge/Data/ODIR_Data/dd/AMD/*.jpg')[0:2]
#path0 = path_list[0]
#img = cv2.imread(path0)
#plt.imshow(img)
 

#ry, rx = estimate_radius(img)
#resize_scale = r / max(rx, ry)
#w = min(int(rx * resize_scale * 2), r*2)
#h = min(int(ry * resize_scale * 2), r*2)
#img = cv2.resize(img, (0,0), fx=resize_scale, fy=resize_scale)
#
#img = crop_img(img, h, w)
#img.shape
#
#img = subtract_gaussian_blur(img)
#plt.imshow(img)
#
#img1 = remove_outer_circle(img, 0.95, r)
#plt.imshow(img1)

images = list() 
labels = list()
for i in range( len( sub_dir_list ) ):
    label = i
    image_names = os.listdir( dir_path + sub_dir_list[i] )
    for image_path in image_names:
        path = dir_path + sub_dir_list[i] + "/" + image_path
        img = cv2.imread(path)
        gray_eye = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#        gray_eye = cv2.cvtColor(cv2.UMat(img), cv2.COLOR_RGB2GRAY)
        gray_eye[gray_eye > 20.0] = 255.0
        y, x = np.where(gray_eye == 255.0)
        img= img[np.min(y):np.max(y),np.min(x):np.max(x)]
        image = cv2.resize(img, (dimen, dimen))
        
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_orig, s_orig, v_orig = cv2.split(image_hsv) 

        # Brightness
        offset = random.randint(-50, 50)
        v = v_orig.astype('int16')
        v[np.logical_and(v < 255 - offset, v > 0 - offset)] += offset
        v[v >= 255 - offset] = 255
        v[v <= 0 - offset] = 0
        v = v.astype('uint8')
        image_3 = cv2.merge((h_orig, s_orig, v))
        image_3 = cv2.cvtColor(image_3, cv2.COLOR_HSV2BGR)
        cv2.imwrite(str(save_path+"Brightness_" +image_path), image_3)


        # Saturation
        offset = random.randint(-50, 50) 
        s = s_orig.astype('int16')
        s[np.logical_and(s < 255 - offset, s > 0 - offset)] += offset
        s[s >= 255 - offset] = 255
        s[s <= 0 - offset] = 0
        s = s.astype('uint8')
        image_4 = cv2.merge((h_orig, s, v_orig))
        image_4 = cv2.cvtColor(image_4, cv2.COLOR_HSV2BGR)
        
        cv2.imwrite(str(save_path+"Saturation" +image_path), image_4)

        # Hue
        offset = random.randint(-15, 15)
        h = h_orig.astype('int16')
        h[np.logical_and(h < 255 - offset, s > 0 - offset)] += offset
        h[h >= 255 - offset] = 255
        h[h <= 0 - offset] = 0
        h = h.astype('uint8')
        image_5 = cv2.merge((h, s_orig, v_orig))
        image_5 = cv2.cvtColor(image_5, cv2.COLOR_HSV2BGR)
        cv2.imwrite(str(save_path+"Hue" +image_path), image_5)
        
        
#        cv2.imwrite(save_path+image_path, resized_image)


 



 

#file = []
#for file in len(label_name):
#    print(file)
#
#for i in range(len(files)):


#
#modified_image_name_list  = image_name_list[np.where(Retinopathy_grade==class_of_interest)]
#
## add additional data 
#	if dd.ix[i] == 1:
#		# add code toresize image
#		for img_path in additional_train_path:
#			src = additional_data_path + '/'+ img_path
#			dstn = modified_train_path + '/'+ img_path
#			# for additional train
#			shrink_image(src).save(dstn)
#
#		for img_path in additional_valid_path:
#			src = additional_data_path + '/'+ img_path
#			dstn = modified_valid_path + '/'+ img_path
#			# for additional valid
#			shrink_image(src).save(dstn)
#
#		for img_path in additional_test_path:
#			src = additional_data_path + '/'+ img_path
#			dstn = modified_test_path + '/'+ img_path
#			# for additional test
#			shrink_image(src).save(dstn)
#
#		print (len(modified_image_name_list) + len(additional_images))
#
#        
        
        
        
        