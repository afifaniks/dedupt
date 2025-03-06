import argparse
from time import time
from typing import List

from data.buckets.bucket_data import BucketData, OtherBucketData
from issue_sim import classic_issues, neural_issues

all_methods = [
    "s3m",
    "transformer",
    "lerch",
    "tracesim",
    "durfex",
    "moroo",
    "rebucket",
    "cosine",
    "levenshtein",
    "brodie",
    "prefix",
]


def classic_issue(data: BucketData, methods: List[str] = None, trim_len: int = 0):
    methods = methods or ["lerch"]
    for method in methods:
        print("Method:", method)
        classic_issues(data, method, max_len=None, trim_len=trim_len)
        print()


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
    parser.add_argument(
        "--encoder_path",
        type=str,
        default=None,
        help="Path to the trained SBERT encoder",
    )
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
    args = parser.parse_args()

    print("Arguments:", args)

    start = time()

    warmup_days = 350
    test_days = 700
    val_days = 140
    train_days = 3850

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

    if args.method == "s3m" or args.method == "transformer":
        neural_issues(
            bucket_netbeans,
            max_len=None,
            trim_len=args.trim_len,
            loss_name=args.loss,
            epochs=4,
            method_name=args.method,
            lang=args.lang,
            multi_stack=args.multi_stack,
            encoder_path=args.encoder_path,
            skip_training=args.skip_training,
            max_frames=max_frames,
        )
    else:
        if args.method == "durfex":
            trim_len = 2
        else:
            trim_len = 0
        classic_issue(bucket_netbeans, [args.method], trim_len=trim_len)

    print("Time:", time() - start)


if __name__ == "__main__":
    main()
