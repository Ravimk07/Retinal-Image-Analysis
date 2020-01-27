import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import random
import argparse 
import os
import numpy as np
import tensorflow as tf 
# this function is converting color class label to gray and define class labels

def bgr2Label(im_label,n):
    if n>2:
        gray = cv2.cvtColor(im_label, cv2.COLOR_BGR2GRAY)
        label_im  = gray.astype("uint8")
    else:
        label_im  = im_label.astype("uint8")
    '''
    #show the image to know gray index value corresponding classes; 
    
    Example: black=0, blue=80, brown=146, gray=135, green=154, pink=159, red=92, yellow=195, white=255
    
    #bv=np.unique(im)
    #print(bv)
    #cv2.imshow('gray_image',im) 
    #cv2.waitKey(0) 
    Please define corresponding classes gray values 
    '''
    gray_val=[92,195,159,135,154,80,91,146,0,255] 
    
    # gray scale values of the gray image which are obtained by unique command (bv) 
 
    im_red= gray_val[0]
    im_yellow= gray_val[1]
    im_pink= gray_val[2]
    im_gray= gray_val[3]
    im_green = gray_val[4]
    im_blue = gray_val[5]
    im_purple = gray_val[6]
    im_brown = gray_val[7]
    im_black = gray_val[8]
    im_white = gray_val[9]
    
    label_im= np.where(label_im==im_black,0,label_im)
    label_im= np.where(label_im==im_white,1,label_im)
    label_im= np.where(label_im==im_red,0,label_im)
    label_im= np.where(label_im==im_yellow,0,label_im)
    label_im= np.where(label_im==im_pink,0,label_im)
    label_im= np.where(label_im==im_gray,0,label_im)
    label_im= np.where(label_im==im_green,0,label_im)
    label_im= np.where(label_im==im_blue,0,label_im)
    label_im= np.where(label_im==im_purple,0,label_im)
    label_im= np.where(label_im==im_brown,0,label_im)
    
    #print(np.unique(im))
    label_im.astype("uint8")  
    return label_im


# image_dir = Path("Z:/Ravi K/Organ/Intestine/Covance_104_data/Layer segmentation/Tiles/Augmented/mask/")
# label_dir = Path("Z:/Ravi K/Organ/Intestine/Covance_104_data/Layer segmentation/Tiles/Augmented/mask/")
# Z:\Ravi K\Other\ISBI_2019_challenge\iChallenge_AMD\Task_3\New_data\exudates\patches\train\augmented\mask
# Z:\Ravi K\Other\ISBI_2019_challenge\iChallenge_AMD\Task_3\Data\Processed\augmented\training\label  masks\testing\ex_label
# Z:\Ravi K\Other\ISBI_2019_challenge\iChallenge_AMD\Task_3\Data\Processed\masks\training\ex_label
image_dir = Path("Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_3/Data/Processed/masks/testing/ex_label/")
label_dir = Path("Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_3/Data/Processed/masks/testing/ex_label/")

# image_dir = Path("Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_2/data/New_final_data/Val/mask/")
# label_dir = Path("Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_2/data/New_final_data/Val/mask/")

affine_en= True        

if image_dir.exists():
    if label_dir.exists():
        image_file_list = [f for f in image_dir.glob("*.png")]
        label_file_list = [f for f in label_dir.glob("*.png")]

        for image_file in image_file_list:
            label_file = label_dir / (image_file.stem + ".png")
            if label_file in label_file_list:
                print(image_file)
#                 print(label_file)

                im_label = cv2.imread(str(image_file)) 
                
                n= len(im_label.shape)
                im_class_labels= bgr2Label(im_label,n)
                
                cv2.imwrite(str(image_dir / (image_file.name)), im_class_labels)

                
            else:
                print("Warning: Label file: {} not present".format(label_file))
                
    else:
        print("Error: Path does not exists: ", os.path.abspath(str(label_dir)))
else:
    print("Error: Path does not exists: ", os.path.abspath(str(image_dir)))

















    
    
