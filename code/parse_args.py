import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # scheduler
    parser.add_argument(
        "--pretrained_scheduler_model_name_or_path",
        type=str,
        default="/hug/models--openai--clip-vit-large-patch14",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    # clip
    parser.add_argument(
        "--pretrained_clip_model_name_or_path",
        type=str,
        default="/hug/models--openai--clip-vit-large-patch14",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default="/hug/models--stabilityai--sd-vae-ft-mse",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_df_model_name_or_path",
        type=str,
        default="/hug/models--SG161222--Realistic_Vision_V4.0_noVAE",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_unet_model_name_or_path",
        type=str,
        default="abc",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    # data_file
    parser.add_argument(
        "--data_json_file",
        type=str,
        default="/proj/wubin/proj/data/train.json",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="/proj/wubin/proj/data",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    # picture size
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    # train_batch_size
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    # tok_k
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    args = parser.parse_args()
    return args