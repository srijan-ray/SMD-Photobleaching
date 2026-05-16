import os
import glob
import numpy as np
import pandas as pd
import ruptures as rpt
from itertools import product
from collections import Counter
from tqdm import tqdm

# ==========================================
# CONFIGURATION - Set your paths here
# ==========================================
input_directory = "/System/Volumes/Data/Volumes/T9/Srijan_KR/labelled_virus/BaL/raw/"  # Directory containing the CSVs
output_file = "Comprehensive_Photobleaching_Results.csv"  # Name of final output file

MAX_BREAKPOINTS = 15
BREAKPOINTS = range(4, MAX_BREAKPOINTS + 1)

# Training parameters
MODELS_WINDOW = ["rbf", "l1"]
WINDOW_SIZES = [8, 10, 12, 15, 18, 20, 22, 25]
MODELS_PELT = ["rbf"]
PENALTIES = np.linspace(0.25, 2.5, 20)
METHODS_OTHER = ["Binseg", "BottomUp", "Dynp"]
MODELS_OTHER = ["rbf", "l1"]
# ==========================================

# 1. Grab all CSV files in the target directory
raw_files = glob.glob(os.path.join(input_directory, "*.txt"))
print(f"Found {len(raw_files)} files. Starting batch processing...")

# List to hold the results dataframe from each file
all_results_dfs = []

# 2. Loop through every file
for fname in tqdm(raw_files, desc="Overall Progress"):
    handle = os.path.basename(fname).split(".")[0]

    # Safely load the data
    try:
        signal = pd.read_csv(fname)
        signal.columns = ["ts", "val"]
        X = signal.val.values
    except Exception as e:
        print(f"Skipping {handle} due to read error: {e}")
        continue

    cps_ls = []  # Reset changepoints for the new file

    print("window method")
    # --- WINDOW METHOD ---
    for m, w, b in product(MODELS_WINDOW, WINDOW_SIZES, BREAKPOINTS):
        cps = rpt.Window(width=w, min_size=2, jump=1).fit_predict(X, n_bkps=b)
        cps_ls.extend(cps[:-1])

    print("pelt method")
    # --- PELT METHOD ---
    for m, p in product(MODELS_PELT, PENALTIES):
        cps = rpt.Pelt(model=m, min_size=2, jump=1).fit_predict(X, pen=p)
        cps_ls.extend(cps[:-1])

    print("other method")
    # --- OTHER METHODS ---
    for method in METHODS_OTHER:
        for m, b in product(MODELS_OTHER, BREAKPOINTS):
            if method == "Binseg":
                print("binseg")
                cps = rpt.Binseg(model=m, min_size=2, jump=1).fit_predict(X, n_bkps=b)
            elif method == "BottomUp":
                print("bottumup")
                cps = rpt.BottomUp(model=m, min_size=2, jump=1).fit_predict(X, n_bkps=b)
            elif method == "Dynp":
                print("dnynp")
                cps = rpt.Dynp(model=m, min_size=2, jump=1).fit_predict(X, n_bkps=b)
            cps_ls.extend(cps[:-1])

    print("aggregating")
    # --- AGGREGATE RESULTS FOR THIS FILE ---
    cps_fq = Counter(cps_ls)
    if not cps_fq:
        continue  # Skip if no changepoints were found at all

    print("dict")
    top_n = dict(cps_fq.most_common(MAX_BREAKPOINTS))
    top_n_keys = sorted(list(top_n.keys())) + [len(signal.ts) - 2]
    scale = max(cps_fq.values())

    results_ls = []
    prev = 0

    print("appending results")
    for ix in top_n_keys:
        # Prevent index out of bounds if ix is right at the end of the array
        if ix >= len(signal.val) - 1:
            ix = len(signal.val) - 2

        results_ls.append(
            [
                handle,  # NEW: Add the file name so you know where this row came from!
                ix,
                signal.ts[ix],
                signal.val[ix - 1],
                signal.val[ix],
                signal.val[ix + 1],
                cps_fq.get(ix, 0) / scale,
                np.mean(signal.val[prev:ix]),
            ]
        )
        prev = ix

    # Convert this file's results into a DataFrame
    file_df = pd.DataFrame(
        results_ls,
        columns=[
            "Source_File",
            "Id",
            "Timestamp",
            "Prev Signal Value",
            "Changepoint Value",
            "Next Signal Value",
            "Changepoint Significance",
            "Average Signal Over Window",
        ],
    )

    # Append to our master list
    print("finish")
    all_results_dfs.append(file_df)

# 3. Combine everything and export
if all_results_dfs:
    # Concatenate all individual DataFrames into one large DataFrame
    comprehensive_df = pd.concat(all_results_dfs, ignore_index=True)

    # Save to a single CSV
    comprehensive_df.to_csv(output_file, index=False)
    print(
        f"\nSuccess! Analyzed {len(all_results_dfs)} files. Saved comprehensive results to {output_file}"
    )
else:
    print(
        "\nNo results generated. Check if the directory path is correct and contains CSVs."
    )
