import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import requests
import os
import random
import shutil
import numpy as np


custom_labels = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 
					'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

# convenience expression for automatically determining device
device = (
    "cuda"
    # Device for NVIDIA or AMD GPUs
    if torch.cuda.is_available()
    else "mps"
    # Device for Apple Silicon (Metal Performance Shaders)
    if torch.backends.mps.is_available()
    else "cpu"
)

# load models
image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
model.to(device)

# expects a PIL.Image or torch.Tensor

root_path = '/home/apal/Documents/ffhq/images1024x1024/'
output_path = 'output'
if not os.path.exists(output_path):
	os.makedirs(output_path)
	
folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
sorted_folders = sorted([f for f in folders if f.isdigit()], key=lambda x: int(x))

idx = 0
numImages = 1
numFolders = len(sorted_folders)
folder_path = root_path + sorted_folders[idx]
all_images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
random_images = random.sample(all_images, numImages)

#url = "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6"
for i in range(numImages):
    image = Image.open(folder_path + '/' + random_images[i])

    # run inference on image
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

    # resize output to match input image dimensions
    upsampled_logits = nn.functional.interpolate(logits,
                size=image.size[::-1], # H x W
                mode='bilinear',
                align_corners=False)

    # get label masks
    labels = upsampled_logits.argmax(dim=1)[0]

    cmap = plt.cm.get_cmap('tab20_r', 19)
    colors = cmap(np.arange(19))[:,:3]

    # move to CPU to visualize in matplotlib
    labels_viz = labels.cpu().numpy()
    
    
    colored_image = colors[labels_viz]
    colored_image = colored_image
    image_base = np.array(image) / 255

    overlay = image_base * 0.2 + colored_image * 0.8
    overlay = np.clip(overlay, 0, 1)
    
    overlay_img = np.dstack((image, labels_viz))
    print(np.shape(overlay_img))
    np.save(output_path + '/' + all_images[i][-9:-4] + '.npy', overlay_img)  
    # plt.imshow(image)

    
    custom_cmap = ListedColormap(colors)
    cbar_array = np.linspace(0,18,19).reshape(19,1)
    cbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=18)),
    ticks=range(19), shrink=1)
    
    cbar.set_ticks(range(19))
    cbar.set_ticklabels(custom_labels)
    
    cbar.set_label("Label", rotation=270, labelpad=20)
    
    # plt.savefig(output_path + '/output_' + random_images[i])
    
    plt.show()
    
    # copy input image over
    # image_input = folder_path + '/' + random_images[i]
    # image_copy = (os.getcwd() + '/output/' + random_images[i])
  
    # shutil.copy(image_input, image_copy)
