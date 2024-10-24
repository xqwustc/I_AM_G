import argparse
import json
import glob
import os
# import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch.nn as nn

import sys
sys.path.append('personalized')
import conf

sys.path.append(os.path.dirname(conf.DIS_PATH))
sys.path.append(os.path.dirname(os.path.dirname(conf.DIS_PATH)))
sys.path.append(conf.DIS_PATH)
print(sys.path)

from DIS import Saliency_ISNET_Node
from skimage import io
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from tqdm import tqdm

class ImageForeground(nn.Module):
    def __init__(self,
                 image_root_path,
                 image_output_path,
                 size=512,
                 ):
        super().__init__()
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.image_root_path = image_root_path
        self.image_output_path = image_output_path

        if not os.path.exists(image_output_path):
            os.makedirs(image_output_path)

        # foreground model
        self.DIS_model = Saliency_ISNET_Node(conf.DIS_PATH, device=device, model_name="isnet-general-use")



    def __call__(self, **kwargs):
        # get path image
        image_glob = os.path.join(self.image_root_path, "*.jpg")
        image_name_list = []
        image_name_list.extend(glob.glob(image_glob))

        for img_path in tqdm(image_name_list, desc="Extracting foregrounds"):
            filename = os.path.basename(img_path)
            des_path = os.path.join(self.image_output_path, filename)  # 输出路径
            self.get_foreground(img_path, des_path)

        print(f"All images processed. Check the output at: {self.image_output_path}")



    def get_foreground(self, img_path, des_path):
        img = io.imread(img_path)
        # FIXME: res return a tuple, thus we only need the first element
        res, _ = self.DIS_model(img)

        saliency_mask = res.astype(np.float32) / 255.  # Convert to a range of 0-1
        segmented_fg = np.expand_dims(saliency_mask, 2) * img.astype(np.float32)  # Use the mask to remove the background
        image = segmented_fg.astype(np.uint8)
        io.imsave(des_path, image)
        return des_path



def main():
    img_processor = ImageForeground(image_root_path=conf.MOVIELENS_IMAGE_PATH, image_output_path=conf.MOVIELENS_FOREGROUND_PATH)
    img_processor()

if __name__ == '__main__':
    main()