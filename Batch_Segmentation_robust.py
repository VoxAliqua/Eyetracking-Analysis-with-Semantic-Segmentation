#%%
# System libs
import os, cv2, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
from cv2 import displayOverlay
from scipy.io import loadmat
from PIL.Image import Image
from IPython.display import display
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle as pkl
import pandas as pd

# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
from types import SimpleNamespace



#%% Create Colourmap
colors = scipy.io.loadmat(r'C:\Users\hanna\CSAIL\Semantic Segmentation\data\color150.mat')['colors']
names = {}
with open(r"C:\Users\hanna\CSAIL\Semantic Segmentation\data\object150_info.csv") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]
    # print(names)
    # print(names[1])

categories = list(names.values())
categories = ['null'] + categories


#%% Options for 
options = SimpleNamespace(fc_dim=2048,
                          num_class=150,
                          imgSizes = [300, 400, 500, 600],
                          imgMaxSize=1000,
                          padding_constant=8,
                          segm_downsampling_rate=8)

# Pre-trained Files
ENCODER_NAME = 'resnet50dilated'    # encoder_epoch_20.pth
DECODER_NAME = 'ppm_deepsup'        # decoder_epoch_20.pth

# ENCODER_NAME = 'resnet101'        # encoder_epoch_50.pth
# DECODER_NAME = 'upernet'          # decoder_epoch_50.pth

pretrained_encoder_file = (r"C:\Users\hanna\CSAIL\Semantic Segmentation\encoder_epoch_20.pth")
pretrained_decoder_file = (r"C:\Users\hanna\CSAIL\Semantic Segmentation\decoder_epoch_20.pth")

# create the model
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

crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module = segmentation_module.eval()
torch.set_grad_enabled(True)

if torch.cuda.is_available():
  segmentation_module = segmentation_module.cuda()

def visualize_result(img, pred, index=None, save=False, prediction=False):
    # (img, info) = data
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        print(f'{names[index+1]}:')

    # # print predictions in descending order
    if prediction is True:
        pred = np.int32(pred)
        pixs = pred.size
        uniques, counts = np.unique(pred, return_counts=True)
        
        for idx in np.argsort(counts)[::-1]:
            name = names[uniques[idx] + 1]
            ratio = counts[idx] / pixs * 100
            if ratio > 0.25:
                print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color),
                            ).astype(np.uint8)
    image = PIL.Image.fromarray(im_vis)
    #Image.show(image)

    if save is True:
        image.save(os.path.join(output_dir, 
                                    f"{film}_mask_{filename[-8:-4]}.png"),
                                    format='png')
    return pred, pred_color


#%% Load and normalize one image as a singleton tensor batch


# input_dir = (r"E:\University\Master\4.SS22\IMaC Lab\Shamem\EyeTracking\Data\Semantic_00_Frames\Downton_Abbey")
# output_dir = (r"E:\University\Master\4.SS22\IMaC Lab\Shamem\EyeTracking\Data\Semantic_01_Segmentation\Robust_batch")
# film = 'Downton_Abbey' #Name
   

# pil_to_tensor = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(
#         mean=[0.34521529, 0.29218697, 0.24947659], # These are RGB mean+std values
#         std=[0.25100187, 0.24346559, 0.23366567])  # across a large photo dataset.
# ])

# pil_image = PIL.Image.open(r"D:\Hannah\University\Master\4.SS22\IMaC Lab\Shamem\EyeTracking\Segmentation\Frames\Frames_Downton_Abbey\frame_00000.jpg").convert('RGB')
# img_original = np.array(pil_image)
# img_data = pil_to_tensor(pil_image)
# singleton_batch = {'img_data': img_data[None].cuda()}
# output_size = img_data.shape[1:]

# ### Run the segmentation at the highest resolution.
# with torch.no_grad():
#     scores = segmentation_module(singleton_batch, segSize=output_size)
    

# ### Get the predicted scores for each pixel
# _, pred = torch.max(scores, dim=1)
# pred = pred.cpu()[0].numpy()
# print(type(pred))


# ### Visualise results
# visualize_result(img_original, pred, index=None, save=False, prediction=True)

# ### Save Labels
# # output_label = pred
# # df_output_label = pd.DataFrame(output_label) #convert to a dataframe
# # df_output_label.to_csv((os.path.join(output_dir, 
# #                                      f"frame_{filename[:-4]}.csv")),
# #                                      sep=';', index=True)

# ### Top classes in answer
# predicted_classes = numpy.bincount(pred.flatten()).argsort()[::-1]
# print(predicted_classes)
# print(type(predicted_classes))

# for c in predicted_classes[:15]:
#     visualize_result(img_original, pred, c)
#     print(pred, c, categories[c])


 #%%Run Multiple files 
import winsound
import time

### Load and normalize one image as a singleton tensor batch
pil_to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
        std=[0.229, 0.224, 0.225])  # across a large photo dataset.
])

input_dir = (r"E:\University\Master\4.SS22\IMaC Lab\Shamem\EyeTracking\Data\Semantic_00_Frames\Downton_Abbey")
output_dir = (r"E:\University\Master\4.SS22\IMaC Lab\Shamem\EyeTracking\Data\Semantic_01_Segmentation\Robust_batch\Downton_Abbey_Masks")
film = 'Downton_Abbey' #Name



t = time.time()     

# Loop over each image in the input directory
for filename in os.listdir(input_dir):
         
    # Load the image
    input_path = os.path.join(input_dir, filename)
    pil_image = PIL.Image.open(input_path).convert('RGB')
    print(f"Processing {input_path}...")
    img_original = np.array(pil_image)

    # Apply the transformation and create a tensor batch
    img_data = pil_to_tensor(pil_image)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]

    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)

    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy() #convert to numpy.array

    #Visualise 
    visualize_result(img_original, pred, index=None, save=True, prediction=False)

    # Save the output_Label
    object_label = pred + 1
    df_object_label = pd.DataFrame(object_label) #convert to a dataframe
    df_object_label.to_csv((os.path.join(output_dir, 
                                                f"{film}_frame_{filename[-8:-4]}.csv")),
                                                sep=';', index=True)


print("executed in %.2fs" % (time.time()-t))
winsound.MessageBeep()   
 


# %%
