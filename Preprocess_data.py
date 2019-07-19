# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:22:20 2019

@author: E75849
"""

import pandas as pd 
import numpy as np
import os
from sklearn.utils import shuffle
import shutil
from PIL import Image
import __future__
 
import glob
import cv2
import numpy as np
import os 
from skimage.io import imread

#-----------------------------------------------------------------
# Grade 0 : 168	Grade 0 : 134	Grade 0 : 34	
# Grade 1 : 25		Grade 1 : 20	Grade 1 : 05	
# Grade 2 : 168	Grade 2 : 136	Grade 2 : 32	
# Grade 3 : 93		Grade 3 : 74	Grade 3 : 19	 	 	 
# Grade 4 : 62		Grade 4 : 49	Grade 4 : 13
#------------------------------------------------------------------
# additional_images Grade 1 = 89 ; split 65 training : 24 testing
# Total images for training
# Class 0: 134
# Class 1: 85 
# Class 2: 136
# Class 3: 74 
# Class 4: 49
# Class 5: 134
# Class 6: 85 
# Class 7: 136
#-------------------------------------------------------------------
# Max reshape size = (1024, 1024)

original_full_data_path ='D:/ODIR_2019 Challenge/Data/ODIR_Data/ODIR-5K_Training_Dataset'

csv_path = pd.read_csv('D:/ODIR_2019 Challenge/Data/ODIR_Data/ODIR-5K_Training_Annotations_R.csv')

label_name = csv_path.iloc[:,4]
image_name = csv_path.iloc[:,3]
#All_label = csv_path.iloc[:,7:15]
#
#print(All_label.iloc[2:4]) glaucoma 

path2write = 'D:/ODIR_2019 Challenge/Data/ODIR_Data/Classes_R/'

file = []
for file in range(len(label_name)):
  if label_name[file]=='cataract':   
     im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
     img = cv2.imwrite(path2write+'Cataract/'+image_name[file],im)
     
  if label_name[file]=='cataract,lens dust':   
     im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
     img = cv2.imwrite(path2write+'Cataract/'+image_name[file],im)

     
  if label_name[file]=='normal fundus':   
     im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
     img = cv2.imwrite(path2write+'Normal/'+image_name[file],im)
     
  if label_name[file]=='lens dust,normal fundus':   
     im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
     img = cv2.imwrite(path2write+'Normal/'+image_name[file],im)
     
  if label_name[file]=='normal fundus,lens dust':   
     im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
     img = cv2.imwrite(path2write+'Normal/'+image_name[file],im)     
     
  if label_name[file]=='lens dust':   
     im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
     img = cv2.imwrite(path2write+'Normal/'+image_name[file],im)
     
  if label_name[file]=='glaucoma':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'Glaucoma/'+image_name[file],im)
      
  if label_name[file]=='suspected glaucoma':   
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'Glaucoma/'+image_name[file],im)
      
  if label_name[file]=='dry age-related macular degeneration': 
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'AMD/'+image_name[file],im)
      
  if label_name[file]== 'wet age-related macular degeneration':   
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'AMD/'+image_name[file],im)
      
  if label_name[file]=='hypertensive retinopathy':
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'hypertensive retinopathy/'+image_name[file],im)
      
  if label_name[file]== 'moderate non proliferative retinopathy,hypertensive retinopathy':   
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'hypertensive retinopathy/'+image_name[file],im)
      img = cv2.imwrite(path2write+'DR/'+image_name[file],im)
      
  if label_name[file]== 'mild non proliferative retinopathy,hypertensive retinopathy':   
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'hypertensive retinopathy/'+image_name[file],im)
      img = cv2.imwrite(path2write+'DR/'+image_name[file],im)
      
  if label_name[file]=='moderate non proliferative retinopathy':
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'DR/'+image_name[file],im)
      
  if label_name[file]=='mild non proliferative retinopathy':   
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'DR/'+image_name[file],im)
      
  if label_name[file]=='proliferative diabetic retinopathy':   
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'DR/'+image_name[file],im)
      
  if label_name[file]=='severe nonproliferative retinopathy':             
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'DR/'+image_name[file],im)
      
  if label_name[file]=='pathological myopia':   
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'Myopia/'+image_name[file],im)
      
  if label_name[file]=='vitreous degeneration':   
     im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
     img = cv2.imwrite(path2write+'Other/'+image_name[file],im)
     
  if label_name[file]=='retinal pigmentation':   
     im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
     img = cv2.imwrite(path2write+'Other/'+image_name[file],im)
     
  if label_name[file]=='branch retinal vein occlusion':   
     im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
     img = cv2.imwrite(path2write+'Other/'+image_name[file],im)
     
  if label_name[file]=='spotted membranous change':   
     im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
     img = cv2.imwrite(path2write+'Other/'+image_name[file],im)
     
  if label_name[file]=='myelinated nerve fibers':   
     im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
     img = cv2.imwrite(path2write+'Other/'+image_name[file],im)
     
  if label_name[file]=='macular epiretinal membrane':   
     im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
     img = cv2.imwrite(path2write+'Other/'+image_name[file],im)
     
  if label_name[file]=='epiretinal membrane':   
     im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
     img = cv2.imwrite(path2write+'Other/'+image_name[file],im)
      
  if label_name[file]=='maculopathy':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'Other/'+image_name[file],im)

  if label_name[file]=='drusen':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'Other/'+image_name[file],im)

  if label_name[file]=='laser spot,moderate non proliferative retinopathy':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'Other/'+image_name[file],im)
      img = cv2.imwrite(path2write+'DR/'+image_name[file],im)
 
  if label_name[file]=='refractive media opacity':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'Other/'+image_name[file],im)
      
      
  if label_name[file]=='cataract,moderate non proliferative retinopathy':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'Cataract/'+image_name[file],im)
      img = cv2.imwrite(path2write+'DR/'+image_name[file],im)
      
      
  if label_name[file]=='glaucoma,diabetic retinopathy':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'Glaucoma/'+image_name[file],im)
      img = cv2.imwrite(path2write+'DR/'+image_name[file],im)
      
      
  if label_name[file]=='glaucoma,myopia retinopathy':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'Glaucoma/'+image_name[file],im)
      img = cv2.imwrite(path2write+'Myopia/'+image_name[file],im)
   
  if label_name[file]=='dry age-related macular degeneration,glaucoma':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'Glaucoma/'+image_name[file],im)
      img = cv2.imwrite(path2write+'AMD/'+image_name[file],im)    
      
      
  if label_name[file]=='mild nonproliferative retinopathy':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])      
      img = cv2.imwrite(path2write+'DR/'+image_name[file],im)
#      img = cv2.imwrite(path2write+'Myopia/'+image_name[file],im)  
      
  if label_name[file]=='tessellated fundus':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])      
      img = cv2.imwrite(path2write+'Other/'+image_name[file],im)

  if label_name[file]=='chorioretinal atrophy':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])      
      img = cv2.imwrite(path2write+'Other/'+image_name[file],im)     
      
  if label_name[file]=='epiretinal membrane,normal fundus':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])      
      img = cv2.imwrite(path2write+'Other/'+image_name[file],im)   
      
  if label_name[file]=='glaucoma,macular epiretinal membrane':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])      
      img = cv2.imwrite(path2write+'Other/'+image_name[file],im)  
      img = cv2.imwrite(path2write+'Glaucoma/'+image_name[file],im)
      
  if label_name[file]=='laser spot,moderate non proliferative retinopathy':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])      
      img = cv2.imwrite(path2write+'Other/'+image_name[file],im)
      img = cv2.imwrite(path2write+'DR/'+image_name[file],im)  
      
  if label_name[file]=='retinochoroidal coloboma':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])      
      img = cv2.imwrite(path2write+'Other/'+image_name[file],im) 
  if label_name[file]=='tessellated fundus,peripapillary atrophy':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])      
      img = cv2.imwrite(path2write+'Other/'+image_name[file],im)       
      












#
#file = []
for file in range(len(label_name)):
    if label_name[file]=='lens dust,myelinated nerve fibers':    
      im = cv2.imread(original_full_data_path +'/'+ image_name[file])     
      img = cv2.imwrite(path2write+'Other/'+image_name[file],im)
#      img = cv2.imwrite(path2write+'DR/'+image_name[file],im)





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
        
        
        
        