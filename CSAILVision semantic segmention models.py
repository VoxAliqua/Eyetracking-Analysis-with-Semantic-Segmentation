#%%
import time
import os
import urllib.request
import cv2, csv
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import scipy.io
import torch
import PIL
from cv2 import displayOverlay
from os.path import exists, join, basename, splitext


torch.cuda.is_available()

# %%

pretrained_encoder_file = (r"C:\Users\hanna\CSAIL\encoder_epoch_50.pth")
pretrained_decoder_file = (r"C:\Users\hanna\CSAIL\decoder_epoch_50.pth")

ENCODER_NAME = 'resnet101'
DECODER_NAME = 'upernet'
# %%

from types import SimpleNamespace
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.dataset import TestDataset
from mit_semseg.utils import colorEncode
from scipy.io import loadmat

colors = scipy.io.loadmat('data/color150.mat')['colors']
names = {}
with open(r"C:\Users\hanna\CSAIL\data\object150_info.csv") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

# options
options = SimpleNamespace(fc_dim=2048,
                          num_class=150,
                          imgSizes = [300, 400, 500, 600],
                          imgMaxSize=1000,
                          padding_constant=8,
                          segm_downsampling_rate=8)

# create model
builder = ModelBuilder()
net_encoder = ModelBuilder.build_encoder(
    arch = ENCODER_NAME, 
    weights=pretrained_encoder_file,
    fc_dim=options.fc_dim)

net_decoder = ModelBuilder.build_decoder(
   arch=DECODER_NAME, 
   weights=pretrained_decoder_file,
   fc_dim=options.fc_dim, 
   num_class=options.num_class, 
   use_softmax=True)

crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module = segmentation_module.eval()
torch.set_grad_enabled(False)

if torch.cuda.is_available():
  segmentation_module = segmentation_module.cuda()

# %%
# test on a given image
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
    # return pred.squeeze(0).cpu().np()

    # Get the predicted segmentation mask as a numpy array
    pred_mask = pred.squeeze(0)
    print(pred_mask)

    return pred_mask, scores

def visualize_result(img, pred, index=None):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        print(f'{names[index+1]}:')
        
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)
        
    return pred, pred_color


# %%
image_path = (r"C:\Users\hanna\CSAIL\input\Images\204.png")

# plt.figure(figsize=(10, 5))
# plt.imshow(matplotlib.image.imread(image_file))

t = time.time()
pred = test(image_path)
print("executed in %.3fs" % (time.time()-t))
print((pred))

# Run the segmentation at the highest resolution.
with torch.no_grad():
    scores = segmentation_module(options.num_class, segSize=output_size)

# Get the predicted scores for each pixel
_, pred = torch.max(scores, dim=1)
pred = pred.cpu()[0].numpy()
visualize_result(img_original, pred)

# pred_color = colorEncode(pred, colors).astype(np.uint8)
# plt.figure(figsize=(10, 5))
# plt.imshow(pred_color)


# fig = plt.figure(figsize=(20, 10))
# ax1 = fig.add_subplot(1, 2, 1)
# ax1.imshow(matplotlib.image.imread(image_file))
# ax2 = fig.add_subplot(1, 2, 2)
# ax2.imshow(pred_color)
# plt.show()


