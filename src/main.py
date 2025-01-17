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
    parser.add_argument("--lang", type=str, help="java/cpp")
    parser.add_argument(
        "--multi_stack", type=bool, default=False, help="Multi stack status"
    )
    args = parser.parse_args()

    start = time()

    bucket_netbeans = OtherBucketData(
        args.bucket_name, args.data_path, 3850, 700, 350, 140, lang=args.lang
    )

    if args.method == "s3m" or args.method == "transformer":
        neural_issues(
            bucket_netbeans,
            max_len=None,
            trim_len=args.trim_len,
            loss_name="ranknet",
            epochs=4,
            method_name=args.method,
            lang=args.lang,
            multi_stack=args.multi_stack,
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
