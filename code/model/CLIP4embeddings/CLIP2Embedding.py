import sys
sys.path.append('/proj/personalized/')
from conf import *
import argparse
import json
import glob
import os
# import h5py
# import numpy as np
# import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer
import conf
import torch.nn as nn

class IMAGE2CLIP(nn.Module):
    def __init__(self,
                 size=512,
                 pretrained_clip_model_name_or_path="",
                 weight_dtype=torch.float32
                 ):
        super().__init__()
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_clip_model_name_or_path)
        self.clip_image_processor = CLIPImageProcessor()
        self.weight_dtype = weight_dtype
    def __call__(self, image_root_path ,batch_size=20,**kwargs):
        # get path image
        Image_glob = os.path.join(image_root_path, "*.png")
        Image_name_list = []
        Image_name_list.extend(glob.glob(Image_glob))

        embedding_dict = {}
        # to gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_encoder.to(device)

        last_start = 0
        # image embedding
        for image_idx in tqdm(range(last_start, len(Image_name_list), batch_size), desc="load image"):
            raw_images = []
            image_paths = Image_name_list[image_idx:image_idx+batch_size]
            item_ids = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]
            err_images = []
            for image_path in image_paths:
                try:
                    raw_image = Image.open(image_path)  # Check whether can be opened correctly
                    raw_images.append(raw_image)
                except:
                    err_images.append(image_path)
                    continue

            clip_images = self.clip_image_processor(images=raw_images, return_tensors="pt").pixel_values

            clip_images = clip_images.to(device, dtype=self.weight_dtype)
            image_embeds = self.image_encoder(clip_images).image_embeds
            image_embeds = image_embeds.detach().cpu()

            cur_embedding = self.create_item_to_embedding(item_ids, image_embeds)
            embedding_dict.update(cur_embedding)

        # save the embeddings
        save_path = os.path.join(os.path.dirname(image_root_path), 'image_embedding.pth')
        torch.save(embedding_dict, save_path)

        print("save the embeddings successfully in image_embedding.pth")


    def create_item_to_embedding(self, item_ids, embeddings):
        item_to_image_embedding = {}
        for item_id, embedding in zip(item_ids, embeddings):
            item_to_image_embedding[item_id] = embedding
        return item_to_image_embedding

class TEXT2CLIP(nn.Module):
    def __init__(self,
                 size=512,
                 pretrained_clip_model_name_or_path="",
                 dataset = "movielens"
                 ):
        super().__init__()
        self.size = size
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_clip_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_clip_model_name_or_path,subfolder="text_encoder")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_encoder.to(self.device)
        self.dataset = dataset
    def __call__(self, text_path, batch_size = 20, **kwargs):
        data_list = []
        embedding_dict = {}
        if self.dataset == "movielens":
            text_file_name = 'text_descriptions.txt'
        elif self.dataset == "MIND":
            text_file_name = 'text_descriptions.txt'
        elif self.dataset == "POG":
            text_file_name = 'POG_text.txt'
        with open(os.path.join(text_path, text_file_name),
                  'r', encoding='utf-8') as file:
            for line in file:
                # Split each line by a colon to extract the ID and text
                parts = line.split(': ')
                if len(parts) >= 2:
                    ID = parts[0].strip()
                    text = ':'.join(parts[1:]).strip()
                    data_list.append((ID, text))

        last_start = 0
        for text_idx in tqdm(range(last_start, len(data_list), batch_size), desc="loading texts"):
            texts = data_list[text_idx: text_idx + batch_size]
            # ONLY TAKE PLACEHOLDER TEXT FOR NOW
            text_ids = [text[0] for text in texts]
            text = [text[1] for text in texts]

            text_input_ids = self.tokenizer(
                text,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)


            # Each item is a (77, 768) tensor
            with torch.no_grad():
                text_embeds = self.text_encoder(text_input_ids)[0].detach().cpu()

            cur_embedding = self.create_item_to_embedding(text_ids, text_embeds)
            embedding_dict.update(cur_embedding)

        # save the embeddings
        save_path = os.path.join(text_path, 'text_embedding.pth')
        torch.save(embedding_dict, save_path)

        print("save the embeddings successfully in text_embedding.json")


    def create_item_to_embedding(self, item_ids, embeddings):
        item_to_image_embedding = {}
        assert len(item_ids) == embeddings.shape[0], 'Number of embeddings and IDS do not match.'
        for i in range(len(item_ids)):
            item_id = item_ids[i]
            embedding = embeddings[i]
            item_to_image_embedding[item_id] = embedding
        return item_to_image_embedding

def main():
    # Image clip
    clip_emb_retrieval_image = IMAGE2CLIP(pretrained_clip_model_name_or_path=conf.IMAGE_ENCODER_PATH)
    clip_emb_retrieval_image(image_root_path=conf.POG_PATH)

    # The use of text clip is similar to image clip

if __name__ == '__main__':
    main()