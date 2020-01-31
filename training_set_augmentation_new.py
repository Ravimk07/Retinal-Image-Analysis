
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
from matplotlib import pyplot as plt
from keras.preprocessing import image
import Augmentor

# image_dir = Path("Z:/Ravi K/Organ/Intestine/Covance_104_data/Layer segmentation/Tiles/Augmented/mask/")
# label_dir = Path("Z:/Ravi K/Organ/Intestine/Covance_104_data/Layer segmentation/Tiles/Augmented/mask/") label

# image_dir = Path("Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_3/New_data/exudates/new_augmented/val/data/")
# label_dir = Path("Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_3/New_data/exudates/new_augmented/val/label/")

image_dir = Path("Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_3/New_data/hemorrhage/new_augmented/val/data/")
label_dir = Path("Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_3/New_data/hemorrhage/new_augmented/val/data/")

from keras.preprocessing.image import ImageDataGenerator

img_gen = ImageDataGenerator()
# img_gen.apply_transform(args)

# image_dir = Path("Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_2/data/New_final_data/Val/mask/")
# label_dir = Path("Z:/Ravi K/Other/ISBI_2019_challenge/iChallenge_AMD/Task_2/data/New_final_data/Val/mask/")


affine_en= True
# input_size = 1024
# 
# def plot_img_and_mask_transformed(img, mask, img_tr, mask_tr): 
#     fig, axs = plt.subplots(ncols=4, figsize=(16, 4), sharex=True, sharey=True)
#     axs[0].imshow(img)
#     axs[1].imshow(mask[:, :, 0])
#     axs[2].imshow(img_tr)
#     axs[3].imshow(mask_tr[:, :, 0])
#     for ax in axs: 
#         ax.set_xlim(0, input_size) 
#         ax.axis('off')
#     fig.tight_layout()
#     plt.show() 
    
    
# IMAGE_HEIGHT = 1024
# IMAGE_WIDTH = 1024
# 
# def zoom(image):
#     zoom_pix = random.randint(0, 10)
#     zoom_factor = 1 + (10*zoom_pix)/IMAGE_HEIGHT
#     image = cv2.resize(image, None, fx=zoom_factor,
#                        fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
#     top_crop = (image.shape[0] - IMAGE_HEIGHT)//2
#     left_crop = (image.shape[1] - IMAGE_WIDTH)//2
#     image = image[top_crop: top_crop+IMAGE_HEIGHT,
#                   left_crop: left_crop+IMAGE_WIDTH]
#     return image

# Change brightness levels
def random_brightness(image):
    # Convert 2 HSV colorspace from BGR colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Generate new random brightness
    rand = random.uniform(0.18, 1.0)
    hsv[:, :, 2] = rand*hsv[:, :, 2]
    # Convert back to BGR colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img
  
from imgaug import augmenters as iaa

# img_rot = rotate(img, theta)
# mask_rot = rotate(mask, theta)
# plot_img_and_mask_transformed(img, mask, img_rot, mask_rot)

  
# img_flip, mask_flip = random_flip(img, mask, u=1)
# plot_img_and_mask_transformed(img, mask, img_flip, mask_flip)               

if image_dir.exists():
    if label_dir.exists():
        image_file_list = [f for f in image_dir.glob("*.jpg")]
        label_file_list = [f for f in label_dir.glob("*.jpg")]

        for image_file in image_file_list:
            label_file = label_dir / (image_file.stem + ".jpg")
            if label_file in label_file_list:
                print(image_file)
#                 print(label_file)
               
                img = cv2.imread(str(image_file))  # for data
#                 label = cv2.imread(str(label_file))
                
                # image = cv2.imread(str(image_file),0)       # for label
#                 label = cv2.imread(str(label_file))
                crop = iaa.Crop(px=(20, 20))
                image_crop = crop.augment_image(img)
                cv2.imwrite(str(image_dir / ("Crop_" + image_file.name)), image_crop)
                
                image_br = random_brightness(img)
                cv2.imwrite(str(image_dir / ("Dar_" + image_file.name)), image_br)
                
                bright = iaa.Add((-20, 20), per_channel=0.5)
                image_bright = bright.augment_image(img)
                cv2.imwrite(str(image_dir / ("Bria_" + image_file.name)), image_bright)
                 
                rotate = iaa.Affine(rotate=(20)) 
                image_aug = rotate.augment_image(img)
                cv2.imwrite(str(image_dir / ("Rot1_" + image_file.name)), image_aug)
                
                rotate = iaa.Affine(rotate=(40))
                image_aug = rotate.augment_image(img)
                cv2.imwrite(str(image_dir / ("Rot2_" + image_file.name)), image_aug)
                
                rotate = iaa.Affine(rotate=(50))
                image_aug = rotate.augment_image(img)
                cv2.imwrite(str(image_dir / ("Rot3_" + image_file.name)), image_aug)
                
                rotate = iaa.Affine(rotate=(-20))
                image_aug = rotate.augment_image(img)
                cv2.imwrite(str(image_dir / ("Rot4_" + image_file.name)), image_aug)
                
                rotate = iaa.Affine(rotate=(-40))
                image_aug = rotate.augment_image(img)
                cv2.imwrite(str(image_dir / ("Rot5_" + image_file.name)), image_aug)

#                 image_zo = zoom(img)
#                 cv2.imwrite(str(image_dir / ("Zo_" + image_file.name)), image_zo)

#                 Blur
                amount = random.uniform(0.5, 1)
                image_2 = cv2.GaussianBlur(img, (5, 5), amount)
                cv2.imwrite(str(image_dir / ("Blur_" + image_file.name)), image_2)

        #        # Flip 
                flipped_imgv = np.fliplr(img)
                cv2.imwrite(str(image_dir / ("flipv_" + image_file.name)), flipped_imgv)
                  
                flipped_imgu = np.flipud(img)
                cv2.imwrite(str(image_dir / ("flipu_" + image_file.name)), flipped_imgu)

                                 
#                 Sharpen = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.25, 0.5))
#                 image_Sharpen = Sharpen.augment_image(img)
#                 cv2.imwrite(str(image_dir / ("sharp_" + image_file.name)), image_Sharpen)

#                 image_6 = image_mirroring(image) 
#                 image_6 = np.fliplr(image)
                
#                 cv2.imwrite(str(image_dir / ("Hue_" + image_file.name)), image_6) 
                
            else:
                print("Warning: Label file: {} not present".format(label_file))
                
    else:
        print("Error: Path does not exists: ", os.path.abspath(str(label_dir)))
else:
    print("Error: Path does not exists: ", os.path.abspath(str(image_dir)))
