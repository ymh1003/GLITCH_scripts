#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate the running times graph")
    p.add_argument("csvfile", nargs="?", help="CSV file",
                   default="figure-experiment.csv")
    p.add_argument("-o", dest="output", help="Output PDF file", default="runningtimes.pdf")
    p.add_argument("--ni", dest="non_interactive", action="store_true")

    args = p.parse_args()

    df = pd.read_csv(args.csvfile)

    outliers = df[df["Ratio"] > 1]
    normal = df[df["Ratio"] <= 1]

    normal_color = 'green'
    outlier_color = 'orange'

    plt.figure(figsize=(6, 6))

    plt.scatter(normal["Print time"], normal["Analysis time"], alpha=0.7,
                color=normal_color)
    plt.scatter(outliers["Print time"], outliers["Analysis time"], color=outlier_color,
                label="Outliers")
    plt.plot([df["Print time"].min(), df["Print time"].max()],
             [df["Print time"].min(), df["Print time"].max()],
             'k--', label='y = x')

    for _, row in outliers.iterrows():
        plt.annotate(row["Benchmark name"], (row["Print time"], row["Analysis time"]),
                     textcoords="offset points", xytext=(5, 5), ha='left', fontsize=9)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Print Time (sec, log scale)")
    plt.ylabel("Analysis Time (sec, log scale)")
    plt.title("Analysis Time vs Print Time (log-log)")
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    plt.tight_layout(w_pad=5)
    plt.savefig(args.output, format="pdf", bbox_inches="tight")
    if not args.non_interactive: plt.show()

