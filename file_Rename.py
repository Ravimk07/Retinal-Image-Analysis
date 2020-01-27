'''
Created on 27-Jun-2019

@author: E75337
File rename
'''

import os
import glob

# data_path= 'Z:/Ravi K/Organ/Intestine/Covance_104_data/Layer segmentation/Tiles/Original/images/'
# path2write= 'Z:/Ravi K/Organ/Intestine/Covance_104_data/Layer segmentation/Tiles/Original/images/' 


data_path= 'Z:/Ravi K/Other/ISBI_2019_challenge/other/diaretdb1_v_1_1/processed/ddb1_fundusimages/'
path2write= 'Z:/Ravi K/Other/ISBI_2019_challenge/other/diaretdb1_v_1_1/processed/ddb1_fundusimages/' 

          
prefix_extension = ''
     
in_directory = os.path.join(data_path)
print (in_directory)
outdirectory = os.path.join(path2write)
if not os.path.exists(outdirectory):
    os.makedirs(outdirectory)
for filename in glob.glob(os.path.join(in_directory)+'/*'):
    print(filename)
    file_name = filename.split('/')
    file_name1 = file_name[-1].split('.')
#     prefix_extension= file_name[-1].split('_BRT')[0]
    file_name2 = filename.split('\\')
    print(file_name2[-1])
    if prefix_extension:
        os.rename(filename,outdirectory+prefix_extension+'_'+file_name2[-1])
    else:
        os.rename(filename,path2write+file_name[0:-4]+file_name[-4:])
