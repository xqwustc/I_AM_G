import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
from utils.logger import logger

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from model.ip_adapter.ip_adapter import ImageProjModel
from model.ip_adapter.utils_ip import is_torch2_available
from conf import *

if is_torch2_available():
    from model.ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from model.ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
from model.RASD import RASD
from collections import Counter
import torch.nn as nn

# Dataset
class CurDataset(torch.utils.data.Dataset):

    def __init__(self, retrieve_augmentor, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05,
                 ti_drop_rate=0.05,dataset = "movielens", image_root_path=""):
        super().__init__()
        self.dataset_name = dataset
        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()
        self.retrieve_augmentor = retrieve_augmentor

        user_history, item_set = self.retrieve_augmentor.get_user_history(dataset)

        self.dataset = dataset

        self.data =  self.from_history_to_data(user_history)
        if image_root_path == "":
            if dataset == "movielens":
                self.image_root_path = MOVIELENS_FOREGROUND_PATH
            elif dataset == "MIND":
                self.image_root_path = MIND_FOREGROUND_PATH
            elif dataset == "POG":
                self.image_root_path = POG_FOREGROUND_PATH


    def from_history_to_data(self, user_history):
        return list(user_history.values())

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.dataset != "POG":
            image_file = item[0] + ".jpg"
        else:
            image_file = item[0] + ".png"

        top_h = 8

        keywords_main = [keyword.strip().lower() for keyword in self.retrieve_augmentor.text_keywords[item[0]]]
        keywords_ax = [self.retrieve_augmentor.text_keywords[i] for i in item[1:]]
        keywords_ax = [keyword.strip().lower() for sublist in keywords_ax for keyword in sublist]

        # 统计每个关键字的出现次数
        keyword_counts = Counter(keywords_ax)

        # 按出现次数从高到低排序
        sorted_keyword_counts = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        top_k_keywords_list = [keyword for keyword, count in sorted_keyword_counts[:top_h]]

        text_origin = self.retrieve_augmentor.text_origin[item[0]]
        # Main keywords
        if self.dataset_name == "movielens":
            text_origin = text_origin + " This movie poster is about " + ", ".join([keyword for keyword in keywords_main])
            text = text_origin + ". "

            # Axuiliary keywords
            text = text + "Maybe related to " + ", ".join([keyword for keyword in top_k_keywords_list])
            text = text + ". "

        elif self.dataset_name == "MIND":
            text_origin = "This news poster is about " + ", ".join([keyword for keyword in keywords_main])
            text = text_origin + ". "

            # Axuiliary keywords
            text = text + "Maybe related to " + ", ".join([keyword for keyword in top_k_keywords_list])
            text = text + "."

        elif self.dataset_name == "POG":
            text_origin = "This outfit is about " + ", ".join([keyword for keyword in keywords_main])
            text = text_origin + ". "

            text = text + "Maybe related to " + ", ".join([keyword for keyword in top_k_keywords_list])
            text = text + "."

        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        image = self.transform(raw_image.convert("RGB"))
        # clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values

        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        img_pos = self.retrieve_augmentor.image_names.index(os.path.splitext(image_file)[0])
        text_pos = self.retrieve_augmentor.text_names.index(os.path.splitext(image_file)[0])

        # FIXME: here are fixed values
        return {
            "ID": item[0],
            "ID_list": item,
            "raw_image": raw_image,
            "image": image,
            "text": text,
            "text_input_ids": text_input_ids,
            "image_kr_embeds": self.retrieve_augmentor.image_embedding[img_pos, :, :, :].unsqueeze(0),
            "text_kr_embeds": self.retrieve_augmentor.text_embedding[text_pos, :, :, :].unsqueeze(0),
            "drop_image_embed": drop_image_embed
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_images = torch.cat([example["image_kr_embeds"] for example in data], dim=0)
    text_kr_embeds = torch.cat([example["text_kr_embeds"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    text = [example["text"] for example in data]

    return {
        "images": images,
        "text": text,
        "text_input_ids": text_input_ids,
        "image_kr_embeds": clip_images,
        "text_kr_embeds": text_kr_embeds,
        "drop_image_embeds": drop_image_embeds
    }


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""

    def __init__(self, unet, image_proj_model, adapter_modules, retrieve_aug = None, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        if retrieve_aug is not None:
            self.retrieve_aug = retrieve_aug

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)
            logger.info("IPAdapter loaded from checkpoint {}".format(ckpt_path))

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds, implicit_embeds): # [batch, 77, 768]  [batch, 768]
        # ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, image_embeds, implicit_embeds], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)


        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        logger.info(f"Successfully loaded weights from checkpoint {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="movielens",
        help=(
            "The dataset to use"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    logger.info("Models loaded!")
    # Freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # ip-adapter
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    ).to('cuda')
    image_proj_model.requires_grad_(False)

    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    dataset_name = args.dataset
    assert dataset_name in ["movielens", "MIND", "POG"], "Invalid dataset name"

    interest_augmentor = RASD(linear_proj=image_proj_model, dataset=dataset_name)
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, retrieve_aug=interest_augmentor,
                           ckpt_path=args.pretrained_ip_adapter_path)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)


    # Optimizer: we only train these two modules
    params_to_opt = itertools.chain(ip_adapter.adapter_modules.parameters(),
                                    interest_augmentor.parameters())

    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    train_dataset = CurDataset(interest_augmentor, tokenizer=tokenizer, size=args.resolution,
                               dataset=dataset_name)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter,
                                                                  optimizer,
                                                                  train_dataloader)

    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(
                        batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    image_kr_embeds = batch["image_kr_embeds"].to(accelerator.device, dtype=weight_dtype)
                    text_kr_embeds = batch["text_kr_embeds"].to(accelerator.device, dtype=weight_dtype)

                image_embeds_ = []
                for image_embed, drop_image_embed in zip(image_kr_embeds[:,0,:,:], batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]

                # Fuse and get the interest embeddings
                implicit_embeds = interest_augmentor(text_kr_embeds, image_kr_embeds)
                noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, image_embeds, implicit_embeds)

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                # Backpropagate
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    logger.info("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}.".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))

            global_step += 1

            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)

            begin = time.perf_counter()


if __name__ == "__main__":
    main()

