import argparse
import glob
import os

from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import proj.conf as conf
from proj.utils.utils import read_json_file, write_json_file


def parse_args():

    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # scheduler
    parser.add_argument(
        "--caption_model_path",
        type=str,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--image_root_path",
        type=str,
        default="proj/data",
        required=False,
        help="image_file_path",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default="16",
        help="",
    )
    parser.add_argument(
        "--train_file_json_path",
        type=str,
        default="train.json",
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
    parser.add_argument(
        "--num_beams",
        type=int,
        default="4",
        help="",
    )
    args = parser.parse_args()
    return args


def process_caption(root_path,json_path,model,feature_extractor,tokenizer,device,gen_kwargs):

    datas = read_json_file(json_path)
    images = []
    for data in datas:
        i_image = Image.open(os.path.join(root_path, data["image_file0"]))
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    for i in range(len(datas)):
        datas[i]["image_caption"] = preds[i]

    write_json_file(datas,json_path)
def main():
    args = parse_args()
    # init caption model
    model = VisionEncoderDecoderModel.from_pretrained(args.caption_model_path)
    feature_extractor = ViTImageProcessor.from_pretrained(args.caption_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.caption_model_path)

    print("info : caption mode have been loaded~~")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gen_kwargs = {"max_length": args.max_length, "num_beams": args.num_beams}
    if args.train_file_json_path != "":
        process_caption(args.image_root_path,args.train_file_json_path,model,feature_extractor,tokenizer,device,gen_kwargs)
    if args.vaild_file_json_path != "":
        process_caption(args.image_root_path,args.vaild_file_json_path,model,feature_extractor,tokenizer,device,gen_kwargs)
    if args.test_file_json_path != "":
        process_caption(args.image_root_path,args.test_file_json_path,model,feature_extractor,tokenizer,device,gen_kwargs)

def get_original_caption(
        batch_size = 20,
    ):
    args = parse_args()
    # load model
    model = VisionEncoderDecoderModel.from_pretrained(conf.CAPTION_MODEL_PATH)
    feature_extractor = ViTImageProcessor.from_pretrained(conf.CAPTION_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(conf.CAPTION_MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gen_kwargs = {"max_length": args.max_length, "num_beams": args.num_beams}

    print("info : caption mode have been loaded~~")
    # get path image
    Image_glob = os.path.join(conf.POG_PATH, "*.png")
    Image_name_list = []
    Image_name_list.extend(glob.glob(Image_glob))


    # image embed
    for image_idx in tqdm(range(0, len(Image_name_list), batch_size), desc="load image"):
        raw_images = []
        image_paths = Image_name_list[image_idx:image_idx + batch_size]
        err_images = []
        for image_path in image_paths:
            try:
                raw_image = Image.open(image_path)  # Check whether can be opened correctly
                if raw_image.mode != "RGB":
                    raw_image = raw_image.convert(mode="RGB")
                raw_images.append(raw_image)
            except:
                err_images.append(image_path)
                continue

        pixel_values = feature_extractor(images=raw_images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        image_paths = [x for x in image_paths if x not in err_images]

        output_ids = model.generate(pixel_values, **gen_kwargs)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]

        # save_Caption :
        with open('caption.csv', 'a') as f:
            for i in range(len(image_paths)):
                f.write(f"{image_paths[i]},{preds[i]}\n")  # write into the file

if __name__ == '__main__':
    # main()
    get_original_caption()