import os
import glob
import pandas as pd
from scipy.stats import wilcoxon
from datetime import datetime

# 1. find the most recent 'exp_*.csv' in results/
exp_files = sorted(glob.glob("tables/exp_*.csv"), key=os.path.getmtime)
if not exp_files:
    raise FileNotFoundError("No exp_*.csv found in tables/")
metrics_csv = exp_files[-1]
print(f"Loading metrics from: {metrics_csv}")
exp_file_tag = os.path.splitext(os.path.basename(metrics_csv))[0]
parsed_file_tage = exp_file_tag.split('_')
exp_tag = '_'.join(parsed_file_tage[0:3]) if len(parsed_file_tage) >= 3 else parsed_file_tage[-1]

# 2. load
df = pd.read_csv(metrics_csv)

# 3. run Wilcoxon tests per dataset, per budget, per metric
metrics = ["accuracy", "f1", "precision", "info_loss"]
sig_recs = []

for ds_name, group_ds in df.groupby("dataset"):
    for budget, group in group_ds.groupby("budget"):
        # pivot so that for each seed you have GA/DAS/SSAS columns
        pivot = group.pivot_table(
            index="seed",
            columns="method",
            values=metrics
        )
        for m in metrics:
            ga_vals   = pivot[m]["GA"].values
            das_vals  = pivot[m]["DAS"].values
            ssas_vals = pivot[m]["SSAS"].values

            # paired Wilcoxon
            stat_das,  p_das  = wilcoxon(ga_vals, das_vals)
            stat_ssas, p_ssas = wilcoxon(ga_vals, ssas_vals)

            sig_recs.append({
                "dataset":       ds_name,
                "budget":        budget,
                "metric":        m,
                "mean_ga":       ga_vals.mean(),
                "std_ga":        ga_vals.std(ddof=0),
                "mean_das":      das_vals.mean(),
                "std_das":       das_vals.std(ddof=0),
                "mean_ssas":     ssas_vals.mean(),
                "std_ssas":      ssas_vals.std(ddof=0),
                "wilcoxon_stat_ga_vs_das":  stat_das,
                "p_ga_vs_das":              p_das,
                "wilcoxon_stat_ga_vs_ssas": stat_ssas,
                "p_ga_vs_ssas":             p_ssas,
            })

sig_df = pd.DataFrame(sig_recs)

# 4. save
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"tables/wilcoxon_per_budget-{exp_tag}.csv"
sig_df.to_csv(out_path, index=False)
print(f"Wilcoxon test results saved to: {out_path}")
