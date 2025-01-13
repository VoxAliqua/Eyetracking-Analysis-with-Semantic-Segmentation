#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import datetime
import argparse
from distutils.version import LooseVersion

# Numerical libs
import cv2
import numpy as np
import torch
import torch.nn as nn
import scipy.io
import csv
import pickle as pkl
import pandas as pd
import PIL.Image, torchvision.transforms
import matplotlib.pyplot as plt

from scipy.io import loadmat
from cv2 import displayOverlay
from scipy.io import loadmat
from PIL.Image import Image
from IPython.display import display

# Specific libs for Algorithm
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from types import SimpleNamespace
from tqdm import tqdm
from mit_semseg.config import cfg


# input_dir = (r"C:\Users\hanna\CSAIL\Semantic Segmentation\input\Images")
# input_file = (r"C:\Users\hanna\CSAIL\Semantic Segmentation\input\Images\114.png")
# output_dir = (r"C:\Users\hanna\CSAIL\Semantic Segmentation\output")


#%%%%%%%%%%%%%%%%%%%%
# Import Colour Map
colours = scipy.io.loadmat(r'C:\Users\hanna\CSAIL\Semantic Segmentation\data\color150.mat')['colors']
names = {}
with open(r"C:\Users\hanna\CSAIL\Semantic Segmentation\data\object150_info.csv") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

categories = list(names.values())
categories = ['null'] + categories

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Creating the Model
ENCODER_NAME = 'resnet101'
DECODER_NAME = 'upernet'

# Path to Encoder File 
pretrained_encoder_file = (r"C:\Users\hanna\CSAIL\Semantic Segmentation\encoder_epoch_50.pth")
pretrained_decoder_file = (r"C:\Users\hanna\CSAIL\Semantic Segmentation\decoder_epoch_50.pth")


options = SimpleNamespace(fc_dim=2048,                                      
                          gpu = 0,
                          num_class=150,
                          imgSizes = [300, 400, 500, 600],
                          imgMaxSize=1000,
                          padding_constant=8,
                          segm_downsampling_rate=8)

### build the model
# Within this pre-trained model set, they seperated the components into an encoder and decoder
    # encoders are usually modified directly from classification networks, 
    # and decoders consist of final convolutions and upsampling.
builder = ModelBuilder()

net_encoder = builder.build_encoder(
    arch=ENCODER_NAME, 
    weights=pretrained_encoder_file,
    fc_dim=options.fc_dim)

net_decoder = builder.build_decoder(
    arch=DECODER_NAME, 
    weights=pretrained_decoder_file,
    fc_dim=options.fc_dim, 
    num_class=options.num_class, 
    use_softmax=True)
 
crit = torch.nn.NLLLoss(ignore_index=-1)                                 # function
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit) # This is the combined momdel 
segmentation_module = segmentation_module.eval()                         # This sets the model to evaluation mode (training and testing being the others)
torch.set_grad_enabled(True)

if torch.cuda.is_available():                                            # Requires a CUDA compatible GPU, highly recommended
  segmentation_module = segmentation_module.cuda()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

# Running the Module
def test(test_image_name):
  dataset_test = TestDataset([{'fpath_img': test_image_name}], options, max_sample=-1)
  
  batch_data = dataset_test[0]
  segSize = (batch_data['img_ori'].shape[0], 
             batch_data['img_ori'].shape[1])
  img_resized_list = batch_data['img_data']
  
  scores = torch.zeros(1, options.num_class, segSize[0], segSize[1])
  if torch.cuda.is_available():
    scores = scores.cuda()

  for img in img_resized_list:
    feed_dict = batch_data.copy()
    feed_dict['img_data'] = img
    del feed_dict['img_ori']
    del feed_dict['info']
    if torch.cuda.is_available():
      feed_dict = {k: o.cuda() for k, o in feed_dict.items()}
      
    
    # forward pass
    pred_tmp = segmentation_module(feed_dict, segSize=segSize)
    scores = scores + pred_tmp / len(options.imgSizes)

    _, pred = torch.max(scores, dim=1)
    return pred.squeeze(0).cpu().numpy()
  



#%%%%%%%%%%%%%%%%%%
# Running Single Image Files

# IMPORTS
# import time
# from PIL import Image
# import numpy as np
# import sys # to access the system
# import cv2

### Directories and Input Pathways ###

# input_dir = (r"C:\Users\hanna\CSAIL\Semantic Segmentation\input\Images")
# input_file = (r"C:\Users\hanna\CSAIL\Semantic Segmentation\input\Images\114.png")
# output_dir = (r"C:\Users\hanna\CSAIL\Semantic Segmentation\output")

# pred = test(input_file)
# predicted_classes = np.bincount(pred.flatten()).argsort()[::-1]
# pred_colour = colorEncode(pred, colours).astype(np.uint8)
# print(type(pred_colour))

# t = time.time()
# print("executed in %.3fs" % (time.time()-t))


# ### print predictions in descending order ###

# pred = np.int32(pred)
# pixs = pred.size
# uniques, counts = np.unique(pred, return_counts=True)
# for idx in np.argsort(counts)[::-1]:
#     name = names[uniques[idx] + 1]
#     ratio = counts[idx] / pixs * 100
#     if ratio > 0.1:
#         print("  {}: {:.2f}%".format(name, ratio))

# ### Save Labels
# output_label = pred
# df_output_label = pd.DataFrame(output_label) #convert to a dataframe
# df_output_label.to_csv((os.path.join(output_dir, 
#                                      f"{input_file[:-4]}.csv")),
#                                      sep=';', index=True) 


# ### Show Images ###
# pil_image = PIL.Image.open(input_file)

# img_original = np.array(pil_image)
# img_mask = PIL.Image.fromarray(pred_colour, 'RGB')
# print(type(img_mask))

# fig = plt.figure(figsize=(15, 10))
# fig.add_subplot(1, 2, 1)
# plt.imshow(img_original)
# fig.add_subplot(1, 2, 2)
# plt.imshow(pred_colour)

# plt.show(block=False)
# plt.pause(5)
# plt.close()





# %%%%Run Multiple files

import winsound
import time

input_dir = (r"E:\University\Master\4.SS22\IMaC Lab\Shamem\EyeTracking\Data\Semantic_00_Frames\Ziemlich_Beste_Freunde")
output_dir = (r"E:\University\Master\4.SS22\IMaC Lab\Shamem\EyeTracking\Data\Semantic_01_Segmentation\Robust_batch\Ziemlich_Beste_Freunde")
film = 'Ziemlich_Beste_Freunde' #Name

t = time.time()


# Loop over each image in the input directory
for filename in os.listdir(input_dir):
    while True:
        try:
            # Load the image
            input_path = os.path.join(input_dir, filename)
            pil_image = PIL.Image.open(input_path)
            img_original = np.array(pil_image)

            print(f"Processing {input_path}...")    
            pred = test(input_path)
            pred_colour = colorEncode(pred, colours).astype(np.uint8)

            #Visualise (optional) 
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(img_original)
            fig.add_subplot(1, 2, 2)
            plt.imshow(pred_colour)
            
            ### Save the Figure
            plt.savefig(os.path.join(output_dir, 
                                    f"{film}_mask_{filename[-8:-4]}.png"),
                                    format='png')
            # plt.show(block=False)
            # plt.pause(0)
            plt.clf()
            plt.close()

            ### Save the object_label
            object_label = pred +1 
            df_object_label = pd.DataFrame(object_label) #convert to a dataframe
            df_object_label.to_csv((os.path.join(output_dir, 
                                                f"{film}_frame_{filename[-8:-4]}.csv")),
                                                sep=';', index=True)
        except Exception:
            continue
        else: 
          break
print("executed in %.2fs" % (time.time()-t))
winsound.MessageBeep()           
    


 

# %%
