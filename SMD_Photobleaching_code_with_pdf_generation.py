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
output_smooth = "Comprehensive_SmoothData.csv"  # NEW: Output for the full step-fits

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

raw_files = glob.glob(os.path.join(input_directory, "*.txt"))
print(f"Found {len(raw_files)} files. Starting batch processing...")

plt.ioff()

all_results_dfs = []
all_overlays_dfs = []  # NEW: Container for the full overlay step data

with pltPdf(output_pdf) as pdfFig:

    for fname in tqdm(raw_files, desc="Overall Progress"):
        handle = os.path.basename(fname).split(".")[0]

        try:
            signal = pd.read_csv(fname, sep="\t")
            signal.columns = ["ts", "val"]
            X = signal.val.values
        except Exception as e:
            print(f"Skipping {handle} due to read error: {e}")
            continue

        cps_ls = []

        # --- RUPTURES METHODS ---
        for m, w, b in product(MODELS_WINDOW, WINDOW_SIZES, BREAKPOINTS):
            cps = rpt.Window(width=w, min_size=2, jump=1).fit_predict(X, n_bkps=b)
            cps_ls.extend(cps[:-1])

        for m, p in product(MODELS_PELT, PENALTIES):
            cps = rpt.Pelt(model=m, min_size=2, jump=1).fit_predict(X, pen=p)
            cps_ls.extend(cps[:-1])

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

        # --- GENERATE PLOTS & SMOOTH DATA OVERLAY ---
        # We only need one high-res overlay array per file, so we generate it
        # based on the max breakpoints (MAX_BREAKPOINTS).
        top_n_plot = dict(cps_fq.most_common(MAX_BREAKPOINTS))

        overlay = list()
        prev_plot = 0
        for ix in sorted(list(top_n_plot.keys())):
            overlay.extend([np.mean(signal.val[prev_plot:ix])] * (ix - prev_plot))
            prev_plot = ix
        overlay.extend([np.mean(signal.val[ix:])] * (len(signal.val) - ix))

        # NEW: Save this file's full high-res step array to our list
        file_overlay_df = pd.DataFrame(
            {
                "Source_File": handle,
                "Timestamp": signal.ts,
                "Original_Signal": signal.val,
                "Step_Fit_Overlay": overlay,
            }
        )
        all_overlays_dfs.append(file_overlay_df)

        # Generate the PDF plots just like before
        for n in BREAKPOINTS:
            top_n_sub = dict(cps_fq.most_common(n))
            plt.figure(figsize=(30, 4))
            plt.title(f"File: {handle} | Significant {n} Change-Points")
            plt.xlabel("Time (sec)")
            plt.ylabel("Intensity (unit/sec)")
            plt.margins(0)

            plt.plot(signal.ts, signal.val, label="Signal")

            sub_overlay = list()
            sub_prev = 0
            for ix in sorted(list(top_n_sub.keys())):
                plt.axvline(x=signal.ts[ix], color="g")
                sub_overlay.extend([np.mean(signal.val[sub_prev:ix])] * (ix - sub_prev))
                sub_prev = ix
            sub_overlay.extend([np.mean(signal.val[ix:])] * (len(signal.val) - ix))

            plt.plot(signal.ts, sub_overlay, color="k", label="Averaged Signal")
            plt.legend()
            pdfFig.savefig()
            plt.close()

# 4. Combine CSV data and export
if all_results_dfs:
    # Export Summary CSV
    comprehensive_df = pd.concat(all_results_dfs, ignore_index=True)
    comprehensive_df.to_csv(output_csv, index=False)

    # Export Smooth Data Overlay CSV
    comprehensive_overlay_df = pd.concat(all_overlays_dfs, ignore_index=True)
    comprehensive_overlay_df.to_csv(output_smooth, index=False)

    print(f"\nSuccess! Analyzed {len(all_results_dfs)} files.")
    print(f"Summary Results saved to: {output_csv}")
    print(f"High-Res Step Fits saved to: {output_smooth}")
    print(f"Plots saved to: {output_pdf}")
else:
    print("\nNo results generated. Check your directory path and file formats.")
