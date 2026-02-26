import argparse
import sys

sys.path.append('/home/mdafifal.mamun/research/S3M')

import json
import sys
from sparse_recommendation import RecommendationFileSparse


p = argparse.ArgumentParser()
p.add_argument("--dedupt_preds", required=True, help="Dedupt JSON result file")
p.add_argument("--other_preds", required=True, help="Other method result file")

args = p.parse_args()

def convert_sparse_to_list(sparse_filepath):
    """
    Convert a .sparse file to python list.
    
    Args:
        sparse_filepath: Path to the input .sparse file
    """
    # Read the sparse file
    sparse_file = RecommendationFileSparse(None, sparse_filepath, -1, only_read=True)
    
    
    results = []
    for query_id, candidates, scores in sparse_file.read_file():
        # Convert to the format expected by RecommendationFileJson
        # Format: (report_id, candidates, scores)
        # Convert numpy types to Python native types for JSON serialization
        candidates_list = [int(c) for c in candidates]
        scores_list = [float(s) for s in scores]
        results.append((int(query_id), candidates_list, scores_list))
    
    return results

other_method_predictions = args.other_preds
dedupt_predictions = args.dedupt_preds

# %%
with open(dedupt_predictions, 'r') as f:
    dedupt_results = json.load(f)

# %%
dedupt_processed_results = []

for _, result in dedupt_results.items():
    bug_id = result["bug_id"]
    actual_duplicate_id = result["actual_duplicate_id"]
    preds = [int(pred) for pred in result["predictions"].keys()]

    dedupt_processed_results.append((bug_id, actual_duplicate_id, preds))
    

# %%
other_method_results = convert_sparse_to_list(other_method_predictions)

# %%
other_method_processed_results = []

for bug_id, candidates, scores in other_method_results:
    other_method_processed_results.append((bug_id, candidates[:25]))

# %%
# Find common bug IDs between the two methods
common_bug_ids = set([result[0] for result in dedupt_processed_results]) & set([result[0] for result in other_method_processed_results])

# Keep only the results for the common bug IDs
dedupt_common_results = [result for result in dedupt_processed_results if result[0] in common_bug_ids]
other_method_common_results = [result for result in other_method_processed_results if result[0] in common_bug_ids]

# %%
len(dedupt_common_results)

# %%
# Create a combined dataframe from the common results
import pandas as pd

combined_results = []

for dedupt_result in dedupt_common_results:
    bug_id, actual_duplicate_id, dedupt_preds = dedupt_result
    
    # Find the corresponding other method result
    other_method_result = next((result for result in other_method_common_results if result[0] == bug_id), None)
    
    if other_method_result is not None:
        _, other_method_preds = other_method_result
        combined_results.append({
            "bug_id": bug_id,
            "actual_duplicate_id": actual_duplicate_id,
            "dedupt_preds": dedupt_preds,
            "other_method_preds": other_method_preds
        })
combined_df = pd.DataFrame(combined_results)

# %%
import pandas as pd

def reciprocal_rank(preds, actual):
    try:
        rank = preds.index(actual) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


combined_df["rr_dedupt"] = combined_df.apply(
    lambda x: reciprocal_rank(x["dedupt_preds"], x["actual_duplicate_id"]),
    axis=1
)

combined_df["rr_other"] = combined_df.apply(
    lambda x: reciprocal_rank(x["other_method_preds"], x["actual_duplicate_id"]),
    axis=1
)


# %%
combined_df

# %%
from scipy.stats import wilcoxon

stat, p_value = wilcoxon(
    combined_df["rr_dedupt"],
    combined_df["rr_other"],
    alternative="two-sided"
)

print(f"Wlicoxon stat: {stat}, P-value: {p_value}")
if p_value < 0.001:
    print(f"    Result: *** Highly significant (p < 0.001)")
elif p_value < 0.01:
    print(f"    Result: ** Very significant (p < 0.01)")
elif p_value < 0.05:
    print(f"    Result: * Significant (p < 0.05)")
else:
    print(f"    Result: Not significant (p >= 0.05)")



