'''
Created on 27-Dec-2019

@author: E75337
'''

# data_path = 'D:/MoNuSAC_images_and_annotations/resize_images/'
# label_path= 'D:/MoNuSAC_images_and_annotations/resize_images/resize_label/'
# path2write_label= 'D:/MoNuSAC_images_and_annotations/resize_images/resize_label/patch_label/'
# path2write_img= 'D:/MoNuSAC_images_and_annotations/resize_images/resize_label/patches/'

data_path = 'Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_3/New_data/exudates/images/val/data/'
label_path = 'Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_3/New_data/exudates/images/val/mask/'

# data_path = 'Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_3/Data/Processed/images/training/scar/'
# label_path = 'Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_3/Data/Processed/masks/training/scar/'

path2write_img= 'Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_3/New_data/exudates/patches/val/data/'
path2write_label= 'Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_3/New_data/exudates/patches/val/mask/'

# path2write='Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_3/Data/Processed/Patches/binary_class/exudate/'

data_extention='.jpg'
label_extention='.png'
import math
import os
import glob 
import cv2
import numpy as np
# os.makedirs(path2write_label)
# os.makedirs(path2write_img)
# height=512
# width=512
classes=2
def binarylabel(im_label,width,height):
    im_label=im_label.astype('uint8')
    lab=np.zeros([width,height,classes],dtype="uint8")
    for i in range(width):
        for j in range(height):
            lab[i,j,im_label[i][j]]=1
    return lab
data=[]   
label=[]
w=[]
h=[]
for filename in glob.glob(label_path+'*'+label_extention):
    #print(filename)
    # Read the label data 
    print(filename)
    pat= filename.split(label_extention)[0]
    im_label = cv2.imread(filename,0)
    print(np.unique(im_label))
    
    pat1=pat.split("\\")[-1]
    im = cv2.imread(data_path+pat1+data_extention)
     
    in_ = np.array(im_label, dtype=np.float32)
    width, height = in_.shape;
    tileSize = 512.0
    rloop = int(math.ceil(width/tileSize))
    cloop = int(math.ceil(height/tileSize))
    out_ = np.zeros((width,height),dtype=np.float32)
#     out_ = np.ones((width,height),dtype=np.float32)
    out_ = out_*255
    for i in range(rloop):
        for j in range(cloop):
            step_i = min(int(tileSize),width-i*int(tileSize))
            step_j = min(int(tileSize),height-j*int(tileSize))
            in_1 = np.zeros((int(tileSize),int(tileSize)),dtype=np.float32)
            in_im = np.zeros((int(tileSize),int(tileSize),3),dtype=np.float32)
            in_1[0:step_i,0:step_j] = in_[i*int(tileSize):i*int(tileSize)+step_i,j*int(tileSize):j*int(tileSize)+step_j]
            cv2.imwrite(path2write_label+pat1+'_'+str(i)+'_'+str(j)+'.png',in_1.astype('uint8'))
             
            in_im[0:step_i,0:step_j,:] = im[i*int(tileSize):i*int(tileSize)+step_i,j*int(tileSize):j*int(tileSize)+step_j,:]
            cv2.imwrite(path2write_img+pat1+'_'+str(i)+'_'+str(j)+'.jpg',in_im.astype('uint8'))
            
#             b_ch=104.00699
#             g_ch=116.66877
#             r_ch=122.67892
#                     
#             im_  = in_im.astype("float32")
#             #Individual channel-wise mean substraction
#             im_ -= np.array((b_ch,g_ch,r_ch))
#              
#             
#     #         # # Compute standard deviation
#     #         b_ch=np.std(im[:,:,0])
#     #         g_ch=np.std(im[:,:,1])
#     #         r_ch=np.std(im[:,:,2])
#     #          
#     #         #Individual channel-wise standard deviation division
#     #         im_ /= np.array((b_ch,g_ch,r_ch))
#              
#         
#             #Append Images into corresponding List
#             data.append(np.rollaxis((im_),2))             
#             #Convert label into binary form
# #             lab = binarylabel(in_1,width,height)
# #             in_1= np.where(in_1<4,0,in_1)
# #             in_1= np.where(in_1>4,0,in_1)
# #             print(np.unique(in_1))
# #             in_1= np.where(in_1==255,1,in_1)
#             print(np.unique(in_1))
#             lab = binarylabel(in_1,1024,1024)
#                  
#             #Append Images into corresponding List
#             label.append(((lab)))
                 
#             print('\n'+tile_name[-1])
#         else:
#             print("error: "+tile_name[-1])
                            
    
# np.save(path2write+data_fileName,np.array(data))
# np.save(path2write+label_fileName,np.array(label))    

#     cv2.imwrite(pat+'.png',51*im_label)
#     print(np.unique(im_label))
#     w.append(im_label.shape[0]) 
#     h.append(im_label.shape[1])
# print(min(w),'and', min(h))
# print(max(w),'and', max(h))


