#!/bin/bash

echo "Start training..."

python train.py \
    --pretrained_model_name_or_path "/hug/models--runwayml--stable-diffusion-v1-5/" \
    --pretrained_ip_adapter_path "/hug/models--h94--IP-Adapter/models/ip-adapter_sd15.bin" \
    --image_encoder_path "/hug/models--h94--IP-Adapter/models/image_encoder"\
    --dataset "movielens"\
    --train_batch_size  16 \
    --save_steps 400