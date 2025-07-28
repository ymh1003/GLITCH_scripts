import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("figure-experiment.csv")

outliers = df[df["Ratio"] > 1]
normal = df[df["Ratio"] <= 1]

normal_color = 'green'
outlier_color = 'orange'

plt.figure(figsize=(12, 5))

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
plt.savefig("benchmark_analysis_plots.pdf", format="pdf", bbox_inches="tight")
plt.show()

