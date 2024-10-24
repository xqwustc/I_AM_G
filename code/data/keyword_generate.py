import argparse
import glob
import os

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

from utils.utils import read_json_file, write_json_file

# init caption model
from transformers import T5Tokenizer, T5ForConditionalGeneration

def parse_args():

    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # scheduler
    parser.add_argument(
        "--keyword_generate_model_path",
        type=str,
        default="/hug/models--Voicelab--vlt5-base-keywords",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        required=False,
        help="",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        required=False,
        help="",
    )
    parser.add_argument(
        "--train_file_json_path",
        type=str,
        default="/proj/wubin/proj/data/train.json",
        required=False,
        help="",
    )
    parser.add_argument(
        "--vaild_file_json_path",
        type=str,
        default="",
        required=False,
        help="",
    )
    parser.add_argument(
        "--test_file_json_path",
        type=str,
        default="",
        required=False,
        help="",
    )
    args = parser.parse_args()
    return args


def keyword_generate(json_path,model,tokenizer,device,no_repeat_ngram_size,num_beams):
    task_prefix = "Keywords: "
    datas = read_json_file(json_path)
    images = []
    for data in datas:
        input_sequences = [task_prefix + data["text0"] + data["image_caption"]]
        input_ids = tokenizer(
            input_sequences, return_tensors="pt", truncation=True
        ).input_ids
        input_ids = input_ids.to(device)
        output = model.generate(input_ids, no_repeat_ngram_size=no_repeat_ngram_size, num_beams=num_beams)
        predicted = tokenizer.decode(output[0], skip_special_tokens=True)

        data["keyword"] = predicted

    write_json_file(datas,json_path)
def main():
    args = parse_args()

    model = T5ForConditionalGeneration.from_pretrained(args.keyword_generate_model_path)
    tokenizer = T5Tokenizer.from_pretrained(args.keyword_generate_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.train_file_json_path != "":
        keyword_generate(args.train_file_json_path,model,tokenizer,device,args.no_repeat_ngram_size,args.num_beams)
    if args.vaild_file_json_path != "":
        keyword_generate(args.vaild_file_json_path,model,tokenizer,device,args.no_repeat_ngram_size,args.num_beams)
    if args.test_file_json_path != "":
        keyword_generate(args.test_file_json_path,model,tokenizer,device,args.no_repeat_ngram_size,args.num_beams)


if __name__ == '__main__':
    main()