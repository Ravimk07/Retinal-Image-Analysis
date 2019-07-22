# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:09:12 2019

@author: E75849
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:00:02 2019

@author: E75849
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:22:20 2019

@author: E75849
"""
 
 

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

path2write ='D:/ODIR_2019 Challenge/Data/ODIR_Data/dd/'
  
def load_resize(base_path, image_ids,  width=400, height=400, resampler_choice=cv2.INTER_AREA, display_images='off'):
	for id in ids:
		path = os.path.join(base_path, id)
		orig_eye_data = np.array(Image.open(path).convert('RGB'))
		gray_img = orig_eye_data[:,:,0]
#
#		gray_eye[gray_eye > 20.0] = 255.0
#
		y, x = np.where(gray_img>20)
#      gray_img = img[:,:,0]
#	   y, x     = np.where(gray_img>20)
		# print np.min(x), np.max(x)

		# remove background 
        
		eye_data = orig_eye_data[np.min(y):np.max(y), np.min(x):np.max(x)]
		resized_image = cv2.resize(eye_data, (width, height))
#      cv2.imwrite(path2write+tileNo[0]+'.png',norm_image)

		
		if display_images == 'None':
#            cv2.imwrite(path2write+tileNo[0]+'.png',norm_image)
			cv2.imwrite(path2write+id,resized_image)
	pass

 
  

path ='D:/ODIR_2019 Challenge/Data/ODIR_Data/dd/AMD'
ids = next(os.walk(path))[2]
load_resize(path, ids, display_images='None')


file_name = 'D:/ODIR_2019 Challenge/Data/ODIR_Data/dd/AMD/102_left.jpg'
image = cv2.imread(file_name)
b,green_fundus,r = cv2.split(image)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

image[:,:,0] = clahe.apply(b)
image[:,:,1] = clahe.apply(green_fundus)
image[:,:,2] = clahe.apply(r)

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(contrast_enhanced_green_fundus)
plt.title('Centered Image')
cv2.imwrite(path2write+'/dd.jpg',image)
            
image = image.transpose(1,2,0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
image[:,:,0] = clahe.apply(image[:,:,0])
image[:,:,1] = clahe.apply(image[:,:,1])
image[:,:,2] = clahe.apply(image[:,:,2])

image = image.transpose(2,0,1)





#
#def shrink_image(img_path):
#	orig_eye_data = Image.open(img_path).convert('RGB')
#
#	img = np.array(orig_eye_data)
#	gray_img = img[:,:,0]
#	y, x     = np.where(gray_img>20)
#	eye_tight= img[np.min(y):np.max(y),np.min(x):np.max(x)]
#	eye_tight= Image.fromarray(eye_tight)
#	return eye_tight
#


        
        
        