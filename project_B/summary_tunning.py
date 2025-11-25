# summary_tuning.py
#
# Summarize results from multiple tuning experiments.
# Assume you have multiple log files such as:
#   results_r10_w1-5_lr2.csv
#   results_r10_w1-8_lr2.csv
#   results_r10_w2-8_lrsmall.csv
#
# Each file has the structure:
#   stage,epoch,model,method,loss,acc,macro_f1,tail_macro_f1
#
# This script will:
#   1. Read all files matching results_*.csv
#   2. Add an experiment name (derived from the file name)
#   3. For each (exp, model, stage), take the best acc/macro_f1/tail_macro_f1
#   4. Save results to summary_tuning.csv and print the table in console

import os
import glob
import pandas as pd


def collect_result_files(pattern: str = "results_*.csv"):
    files = glob.glob(pattern)
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print("No results_*.csv files found.")
    else:
        print("Found result files:")
        for f in files:
            print("  -", f)
    return files


def summarize_results(files, out_path: str = "summary_tuning.csv"):
    all_rows = []

    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

        # Derive experiment name from file name, e.g. results_r10_w1-5_lr2.csv -> r10_w1-5_lr2
        base = os.path.basename(path)
        exp_name = os.path.splitext(base)[0]  # remove .csv extension
        # Remove prefix "results_" if present
        if exp_name.startswith("results_"):
            exp_name = exp_name[len("results_"):]

        df["exp"] = exp_name
        all_rows.append(df)

    if not all_rows:
        print("No valid result data loaded.")
        return

    full_df = pd.concat(all_rows, ignore_index=True)

    # For each (exp, model, stage), take the maximum acc/macro_f1/tail_macro_f1
    group_cols = ["exp", "model", "stage"]
    agg_df = (
        full_df
        .groupby(group_cols)
        .agg({
            "acc": "max",
            "macro_f1": "max",
            "tail_macro_f1": "max",
            "loss": "min",  # smaller loss roughly indicates better performance
        })
        .reset_index()
    )

    # Sort results nicely: stage1 before stage2
    stage_order = {"stage1": 0, "stage2": 1}
    agg_df["stage_order"] = agg_df["stage"].map(stage_order).fillna(99).astype(int)
    agg_df = agg_df.sort_values(["exp", "model", "stage_order"]).drop(columns=["stage_order"])

    agg_df.to_csv(out_path, index=False)
    print(f"\nSummary saved to: {out_path}\n")
    print(agg_df)


if __name__ == "__main__":
    files = collect_result_files("results_*.csv")
    if files:
        summarize_results(files, out_path="summary_tuning.csv")
