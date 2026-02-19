import argparse
import os
from datetime import datetime, timezone

from data.bug_report_database import BugReportDatabase
from util.data_util import read_date_from_report


def save_dataset_file(path, info, reports, duplicate_reports):
    with open(path, "w") as f:
        f.write(info + "\n")
        f.write(" ".join(str(x) for x in reports) + "\n")
        f.write(" ".join(str(x) for x in duplicate_reports) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--database", required=True, help="TraceSim JSON database (same as used in evaluation)")
    p.add_argument("--out_dir", required=True, help="Output folder for chunk_0 files")
    p.add_argument("--warmup_days", type=int, default=0)
    p.add_argument("--train_days", type=int, required=True)
    p.add_argument("--val_days", type=int, default=0)
    p.add_argument("--test_days", type=int, required=True)
    p.add_argument("--prefix", default="", help="Optional prefix for filenames")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    report_db = BugReportDatabase.from_json(args.database)
    master_id_by_report_id = report_db.get_master_by_report()

    # Sort all reports by creation time (and tie-break by bug_id)
    reports = []
    for r in report_db.report_list:
        rid = r["bug_id"]
        ts = read_date_from_report(r).timestamp()
        reports.append((rid, ts))
    reports.sort(key=lambda x: (x[1], x[0]))

    if not reports:
        raise SystemExit("Empty database")

    day_secs = 24 * 60 * 60
    first_day = int(reports[0][1] / day_secs)

    def rel_day(ts: float) -> int:
        return int(ts / day_secs) - first_day

    # DedupT-style boundaries (days since first report)
    warmup_end = args.warmup_days
    train_end = warmup_end + args.train_days
    val_end = train_end + args.val_days
    test_end = val_end + args.test_days

    train_ids, val_ids, test_ids = [], [], []
    train_dup, val_dup, test_dup = [], [], []

    for rid, ts in reports:
        d = rel_day(ts)

        if d < args.warmup_days:
            # warmup: belongs to history, not a query set
            continue

        if warmup_end <= d < train_end:
            train_ids.append(rid)
            if master_id_by_report_id[rid] != rid:
                train_dup.append(rid)
        elif train_end <= d < val_end:
            val_ids.append(rid)
            if master_id_by_report_id[rid] != rid:
                val_dup.append(rid)
        elif val_end <= d < test_end:
            test_ids.append(rid)
            if master_id_by_report_id[rid] != rid:
                test_dup.append(rid)

    # Info string with actual wall-clock bounds (helpful for your paper/rebuttal)
    def day_to_date_str(day_idx: int) -> str:
        # day_idx is relative day since first_day
        abs_day = first_day + day_idx
        return datetime.fromtimestamp(abs_day * day_secs, tz=timezone.utc).strftime("%Y-%m-%d")

    info = (
        f"DedupT-style split; warmup={args.warmup_days} train={args.train_days} "
        f"val={args.val_days} test={args.test_days}; "
        f"train=[{day_to_date_str(warmup_end)},{day_to_date_str(train_end)}) "
        f"val=[{day_to_date_str(train_end)},{day_to_date_str(val_end)}) "
        f"test=[{day_to_date_str(val_end)},{day_to_date_str(test_end)})"
    )

    pref = args.prefix
    if pref and not pref.endswith("_"):
        pref += "_"

    save_dataset_file(os.path.join(args.out_dir, f"{pref}training_chunk_0.txt"), info, train_ids, train_dup)
    save_dataset_file(os.path.join(args.out_dir, f"{pref}validation_chunk_0.txt"), info, val_ids, val_dup)
    save_dataset_file(os.path.join(args.out_dir, f"{pref}test_chunk_0.txt"), info, test_ids, test_dup)

    print("Wrote:")
    print(" ", os.path.join(args.out_dir, f"{pref}training_chunk_0.txt"), f"(n={len(train_ids)}, dup={len(train_dup)})")
    print(" ", os.path.join(args.out_dir, f"{pref}validation_chunk_0.txt"), f"(n={len(val_ids)}, dup={len(val_dup)})")
    print(" ", os.path.join(args.out_dir, f"{pref}test_chunk_0.txt"), f"(n={len(test_ids)}, dup={len(test_dup)})")


if __name__ == "__main__":
    main()

