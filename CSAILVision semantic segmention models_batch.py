from colabcode import ColabCode

import csv
import time
import os
import urllib.request
import matplotlib
import matplotlib.pylab as plt
import numpy as np
from os.path import exists, join, basename, splitext

from types import SimpleNamespace
import torch
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.dataset import TestDataset
from mit_semseg.utils import colorEncode
from scipy.io import loadmat

# %%

ENCODER_NAME = 'resnet101'
DECODER_NAME = 'upernet'

pretrained_encoder_file = (r"C:\Users\hanna\CSAIL\encoder_epoch_50.pth")
pretrained_decoder_file = (r"C:\Users\hanna\CSAIL\decoder_epoch_50.pth")


colors = loadmat('data/color150.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
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
net_encoder = builder.build_encoder(arch=ENCODER_NAME, weights=pretrained_encoder_file,
                                    fc_dim=options.fc_dim)
net_decoder = builder.build_decoder(arch=DECODER_NAME, weights=pretrained_decoder_file,
                                    fc_dim=options.fc_dim, num_class=options.num_class, use_softmax=True)
crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module = segmentation_module.eval()
torch.set_grad_enabled(False)

if torch.cuda.is_available():
  segmentation_module = segmentation_module.cuda()

# %%
# test on a given image

def test(input_folder, output_folder):
  # Create output folder if it doesn't exist
  if not os.path.exists(output_folder):
      os.makedirs(output_folder)
  
  # Get a list of all image files in the input folder
  image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

  for image_file in image_files:
    # Load input image
    test_image_name = os.path.join(input_folder, image_file)
    dataset_test = TestDataset([{'fpath_img': test_image_name}], options, max_sample=-1)
  
    batch_data = dataset_test[0]
    segSize = (batch_data['img_ori'].shape[0], batch_data['img_ori'].shape[1])
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
      
      # Get the predicted segmentation mask as a numpy array
      pred_mask = pred.squeeze(0).gpu().numpy()

      # Save the mask to a file using numpy's savez function
      save_path = os.path.join(output_folder, f'{os.path.splitext(image_file)[0]}.npz')
      np.savez(save_path, pred_mask=pred_mask)
      
      return pred_mask





# %%
input_folder = r"C:\Users\hanna\CSAIL\input\Images"
output_folder = r"C:\Users\hanna\CSAIL\output"


image_file = (r"C:\Users\hanna\CSAIL\input\Images\105.png")

#plt.figure(figsize=(10, 5))
#plt.imshow(matplotlib.image.imread(image_file))



t = time.time()
pred = test(input_folder)
print("executed in %.3fs" % (time.time()-t))

pred_color = colorEncode(pred, loadmat('data/color150.mat')['colors'])
#plt.figure(figsize=(10, 5))
#plt.imshow(pred_color)


fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(matplotlib.image.imread(image_file))
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(pred_color)
plt.show()




# %%
