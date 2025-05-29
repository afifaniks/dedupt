import argparse
from time import time
from typing import List

from dotenv import load_dotenv

from data.buckets.bucket_data import BucketData, OtherBucketData
from issue_sim_llm import neural_issues

load_dotenv()

all_methods = [
    "llm"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, help=f"Method name. One of {all_methods}")
    parser.add_argument(
        "--trim_len",
        type=int,
        default=0,
        required=False,
        help="Trim length for S3M method",
    )
    parser.add_argument("--data_path", type=str, help="Path to file with reports")
    parser.add_argument("--bucket_name", type=str, help="Bucket name of reports")
    parser.add_argument("--lang", type=str, help="java/cpp")
    parser.add_argument(
        "--multi_stack", action="store_true", help="Enable multi stack status"
    )
    parser.add_argument(
        "--skip_training", action="store_true", help="Skip training of the model"
    )
    parser.add_argument(
        "--loss", type=str, default="ranknet", help="Loss function for neural methods"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        required=False,
        help="Maximum frames to be considered",
    )
    parser.add_argument(
        "--train_days",
        type=int,
        default=3500,
        required=False,
        help="Train days for the bucket",
    )
    parser.add_argument(
        "--test_days",
        type=int,
        default=700,
        required=False,
        help="Test days for the bucket",
    )
    parser.add_argument(
        "--warmup_days",
        type=int,
        default=350,
        required=False,
        help="Warmup days for the bucket",
    )
    parser.add_argument(
        "--val_days",
        type=int,
        default=140,
        required=False,
        help="Validation days for the bucket",
    )
    args = parser.parse_args()

    print("Arguments:", args)

    start = time()

    warmup_days = args.warmup_days
    test_days = args.test_days
    val_days = args.val_days
    train_days = args.train_days

    default_frames = {
        "gnome": 100,
        "netbeans": 10,
        "eclipse": 10,
        "ubuntu": 50,
    }

    max_frames = args.max_frames if args.max_frames > 0 else None
    if not max_frames:
        max_frames = default_frames.get(args.bucket_name, 10)
    print("Max frames:", max_frames)

    bucket_netbeans = OtherBucketData(
        args.bucket_name,
        args.data_path,
        train_days,
        test_days,
        warmup_days,
        val_days,
        lang=args.lang,
    )

    if args.method == "llm":
        neural_issues(
            bucket_netbeans,
            max_len=None,
            trim_len=args.trim_len,
            loss_name=args.loss,
            epochs=2,
            method_name=args.method,
            lang=args.lang,
            multi_stack=args.multi_stack,
            encoder_path="",
            skip_training=args.skip_training,
            max_frames=max_frames,
        )
    else:
        raise ValueError(
            f"Method {args.method} is not supported. Supported methods: {all_methods}"
        )

    print("Time:", time() - start)


if __name__ == "__main__":
    main()
