import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import powerlaw, geom, zscore, rankdata
from sklearn.preprocessing import quantile_transform

# Generate Data
np.random.seed(42)  # Ensure reproducibility
B = np.random.normal(5, 2, 10000)
I = powerlaw.rvs(0.3, size=10000)
H = geom.rvs(0.005, size=10000)

# Boxplot for original data
plt.figure(figsize=(12, 6))
sns.boxplot(data=[B, I, H], palette="Set2", linewidth=2, width=0.6)
plt.xticks([0, 1, 2], ['B (Gaussian)', 'I (Power Law)', 'H (Geometric)'], fontsize=12, fontweight='bold')
plt.title('Original Variable Distribution', fontsize=16, fontweight='bold', color='darkred')
plt.grid(True, linestyle='-.', alpha=0.8)
plt.show()

def plot_histograms(original, transformed, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(original, bins=50, color='darkred', kde=True, label='Original', alpha=0.5)
    sns.histplot(transformed, bins=50, color='green', kde=True, label='Transformed', alpha=0.5)
    plt.title(title, fontsize=16, fontweight='bold', color='navy')
    plt.legend()
    plt.grid(True, linestyle='-.', alpha=0.8)
    plt.show()

# Normalization Methods
B_max, I_max, H_max = B / B.max(), I / I.max(), H / H.max()
B_sum, I_sum, H_sum = B / B.sum(), I / I.sum(), H / H.sum()
B_z, I_z, H_z = zscore(B), zscore(I), zscore(H)
B_pct, I_pct, H_pct = rankdata(B) / len(B), rankdata(I) / len(I), rankdata(H) / len(H)

# Adjust medians
medians = [np.median(B), np.median(I), np.median(H)]
m1 = np.mean(medians)
B_med, I_med, H_med = B * (m1 / medians[0]), I * (m1 / medians[1]), H * (m1 / medians[2])

# Quantile Normalization
stacked_data = np.vstack([B, I, H]).T
quantile_norm = quantile_transform(stacked_data, axis=0, copy=True)
B_q, I_q, H_q = quantile_norm.T

# Histogram Comparisons
normalizations = ["Max", "Sum", "Z-Score", "Percentile", "Median Adjusted", "Quantile"]
transformed_data = [(B_max, I_max, H_max), (B_sum, I_sum, H_sum),
                    (B_z, I_z, H_z), (B_pct, I_pct, H_pct),
                    (B_med, I_med, H_med), (B_q, I_q, H_q)]

for name, (B_new, I_new, H_new) in zip(normalizations, transformed_data):
    plot_histograms(B, B_new, f'B - {name} Normalization')
    plot_histograms(I, I_new, f'I - {name} Normalization')
    plot_histograms(H, H_new, f'H - {name} Normalization')
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=[B_new, I_new, H_new], palette="Set1", linewidth=2, width=0.6)
    plt.xticks([0, 1, 2], ['B', 'I', 'H'], fontsize=12, fontweight='bold')
    plt.title(f'Box Plot Comparison - {name} Normalization', fontsize=16, fontweight='bold', color='darkred')
    plt.grid(True, linestyle='-.', alpha=0.8)
    plt.show()
