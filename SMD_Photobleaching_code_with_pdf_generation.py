import os
import glob
import numpy as np
import pandas as pd
import ruptures as rpt
from itertools import product
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages as pltPdf

# ==========================================
# CONFIGURATION - Set your paths here
# ==========================================
input_directory = "./path/to/your/raw_csv_folder"  # Directory containing the TXT files
output_csv = "Comprehensive_Photobleaching_Results.csv"
output_pdf = "Comprehensive_Photobleaching_Plots.pdf"

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

# 1. Grab all TXT files in the target directory
raw_files = glob.glob(os.path.join(input_directory, "*.txt"))
print(f"Found {len(raw_files)} files. Starting batch processing...")

# Turn off interactive plotting so Jupyter doesn't crash from drawing hundreds of plots
plt.ioff()

all_results_dfs = []

# 2. Open the comprehensive PDF file
with pltPdf(output_pdf) as pdfFig:

    # 3. Loop through every file
    for fname in tqdm(raw_files, desc="Overall Progress"):
        handle = os.path.basename(fname).split(".")[0]

        # Safely load the data (adjust sep='\t' or sep=',' depending on your txt format)
        try:
            signal = pd.read_csv(fname, sep="\t")
            signal.columns = ["ts", "val"]
            X = signal.val.values
        except Exception as e:
            print(f"Skipping {handle} due to read error: {e}")
            continue

        cps_ls = []

        # --- WINDOW METHOD ---
        for m, w, b in product(MODELS_WINDOW, WINDOW_SIZES, BREAKPOINTS):
            cps = rpt.Window(width=w, min_size=2, jump=1).fit_predict(X, n_bkps=b)
            cps_ls.extend(cps[:-1])

        # --- PELT METHOD ---
        for m, p in product(MODELS_PELT, PENALTIES):
            cps = rpt.Pelt(model=m, min_size=2, jump=1).fit_predict(X, pen=p)
            cps_ls.extend(cps[:-1])

        # --- OTHER METHODS ---
        for method in METHODS_OTHER:
            for m, b in product(MODELS_OTHER, BREAKPOINTS):
                if method == "Binseg":
                    cps = rpt.Binseg(model=m, min_size=2, jump=1).fit_predict(
                        X, n_bkps=b
                    )
                elif method == "BottomUp":
                    cps = rpt.BottomUp(model=m, min_size=2, jump=1).fit_predict(
                        X, n_bkps=b
                    )
                elif method == "Dynp":
                    cps = rpt.Dynp(model=m, min_size=2, jump=1).fit_predict(X, n_bkps=b)
                cps_ls.extend(cps[:-1])

        # --- AGGREGATE RESULTS ---
        cps_fq = Counter(cps_ls)
        if not cps_fq:
            continue

        top_n = dict(cps_fq.most_common(MAX_BREAKPOINTS))
        top_n_keys = sorted(list(top_n.keys())) + [len(signal.ts) - 2]
        scale = max(cps_fq.values())

        results_ls = []
        prev = 0

        for ix in top_n_keys:
            if ix >= len(signal.val) - 1:
                ix = len(signal.val) - 2

            results_ls.append(
                [
                    handle,
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
        all_results_dfs.append(file_df)

        # --- GENERATE PLOTS FOR THE PDF ---
        for n in BREAKPOINTS:
            # select top N changepoints
            top_n_plot = dict(cps_fq.most_common(n))

            # Create a new figure
            plt.figure(figsize=(30, 4))

            # I added the filename [handle] to the title so you know which file this is!
            plt.title(f"File: {handle} | Significant {n} Change-Points")
            plt.xlabel("Time (sec)")
            plt.ylabel("Intensity (unit/sec)")
            plt.margins(0)

            # plot original signal
            plt.plot(signal.ts, signal.val, label="Signal")

            # overlay averaged signal values
            overlay = list()
            prev_plot = 0
            for ix in sorted(list(top_n_plot.keys())):
                plt.axvline(x=signal.ts[ix], color="g")
                overlay.extend([np.mean(signal.val[prev_plot:ix])] * (ix - prev_plot))
                prev_plot = ix
            overlay.extend([np.mean(signal.val[ix:])] * (len(signal.val) - ix))

            # plot overlay
            plt.plot(signal.ts, overlay, color="k", label="Averaged Signal")
            plt.legend()

            # Save the figure to the comprehensive PDF
            pdfFig.savefig()

            # Close the figure to free up system memory
            plt.close()

# 4. Combine CSV data and export
if all_results_dfs:
    comprehensive_df = pd.concat(all_results_dfs, ignore_index=True)
    comprehensive_df.to_csv(output_csv, index=False)
    print(f"\nSuccess! Analyzed {len(all_results_dfs)} files.")
    print(f"Data saved to: {output_csv}")
    print(f"Plots saved to: {output_pdf}")
else:
    print("\nNo results generated. Check your directory path and file formats.")
