import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, accuracy_score

# Generate heights data.
def generate_heights(mean, std_dev, size, label):
    heights = np.random.normal(mean, std_dev, size)
    labels = [label] * size
    return pd.DataFrame({'height': heights, 'label': labels})

# Plot histograms.
def plot_histograms(female_heights, male_heights, bins=50, title_suffix=""):
    plt.figure()
    plt.hist([female_heights, male_heights], bins=bins, label=['Female', 'Male'],
             alpha=0.7, color=['purple', 'green'])
    plt.title(f'Height Distributions {title_suffix}')
    plt.xlabel('Height (cm)')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

# Threshold classifier.
def threshold_classifier(female_data, male_data, threshold):
    predictions = np.where(np.concatenate([female_data, male_data]) < threshold, 'F', 'M')
    actual = np.concatenate([['F'] * len(female_data), ['M'] * len(male_data)])
    return actual, predictions

# Probability classifier.
def probability_classifier(female_data, male_data, female_mean, female_sd, male_mean, male_sd):
    def classify(height):
        female_prob = norm.pdf(height, female_mean, female_sd)
        male_prob = norm.pdf(height, male_mean, male_sd)
        return 'F' if female_prob > male_prob else 'M'
    predictions = np.array([classify(h) for h in np.concatenate([female_data, male_data])])
    actual = np.concatenate([['F'] * len(female_data), ['M'] * len(male_data)])
    return actual, predictions

# Quantized classifier.
def quantized_classifier(female_data, male_data, interval_len):
    def quantize(data):
        intervals = np.floor(data / interval_len)
        return pd.Series(intervals).value_counts()
    female_quantized = quantize(female_data)
    male_quantized = quantize(male_data)
    quantized_ranges = set(female_quantized.index).union(set(male_quantized.index))
    predictions = []
    actual = []
    for interval in quantized_ranges:
        female_count = female_quantized.get(interval, 0)
        male_count = male_quantized.get(interval, 0)
        majority_label = 'F' if female_count >= male_count else 'M'
        predictions.extend([majority_label] * (female_count + male_count))
        actual.extend(['F'] * female_count + ['M'] * male_count)
    return np.array(actual), np.array(predictions)

# Evaluate classifier.
def evaluate_classifier(actual, predictions, description=""):
    cm = confusion_matrix(actual, predictions, labels=['F', 'M'])
    accuracy = accuracy_score(actual, predictions)
    return cm, accuracy

# Detect outliers using z-score.
def detect_outliers_zscore(data, cutoff=2):
    mean_val = np.mean(data)
    std_val = np.std(data)
    z_scores = (data - mean_val) / std_val
    outlier_indices = np.where(np.abs(z_scores) > cutoff)[0]
    return outlier_indices

# Detect outliers using IQR.
def detect_outliers_iqr(data, factor=1.5):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
    return outlier_indices

# Detect outliers using MAD.
def detect_outliers_mad(data, cutoff=3):
    median_val = np.median(data)
    mad = np.median(np.abs(data - median_val))
    if mad == 0:
        return np.array([])
    modified_z_scores = 0.6745 * (data - median_val) / mad
    outlier_indices = np.where(np.abs(modified_z_scores) > cutoff)[0]
    return outlier_indices

# Main function for Assignment 3.
def main_assignment3():
    female_mean = 152
    male_mean = 166
    sd = 7.5
    sample_size = 1000

    # --- Generate Data ---
    df_female = generate_heights(female_mean, sd, sample_size, 'F')
    df_male = generate_heights(male_mean, sd, sample_size, 'M')
    female_data = df_female['height'].values
    male_data = df_male['height'].values

    # --- Part (a) & (b): Original vs. Altered Data ---
    # Plot overall histograms for comparison.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].hist([female_data, male_data], bins=30, label=['Female', 'Male'],
                 alpha=0.7, color=['purple', 'green'])
    axes[0].set_title("Original Data")
    axes[0].set_xlabel("Height (cm)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    
    # Increase top 50 female heights by 10 cm.
    indices_top50 = np.argsort(female_data)[-50:]
    female_data_altered = female_data.copy()
    female_data_altered[indices_top50] += 10

    axes[1].hist([female_data_altered, male_data], bins=30, label=['Female', 'Male'],
                 alpha=0.7, color=['purple', 'green'])
    axes[1].set_title("Altered Data")
    axes[1].set_xlabel("Height (cm)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    plt.tight_layout()
    plt.show()
    
    print("=== Female Data Statistics ===")
    print("Before alteration: Mean = {:.2f}, SD = {:.2f}".format(np.mean(female_data), np.std(female_data)))
    print("After alteration:  Mean = {:.2f}, SD = {:.2f}".format(np.mean(female_data_altered), np.std(female_data_altered)))
    
    # --- Box Plot Comparison: Original vs. Altered Data ---
    # Create combined DataFrames.
    df_original = pd.concat([df_female, df_male], ignore_index=True)
    df_female_altered = df_female.copy()
    df_female_altered.loc[np.argsort(df_female_altered['height'])[-50:], 'height'] += 10
    df_altered = pd.concat([df_female_altered, df_male], ignore_index=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Modified sns.boxplot calls:
    sns.boxplot(x='label', y='height', hue='label', data=df_original,
                palette={'F':'purple','M':'green'}, dodge=False, ax=axes[0])
    axes[0].set_title("Original Data Box Plot")
    axes[0].set_xlabel("Gender")
    axes[0].set_ylabel("Height (cm)")
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    
    sns.boxplot(x='label', y='height', hue='label', data=df_altered,
                palette={'F':'purple','M':'green'}, dodge=False, ax=axes[1])
    axes[1].set_title("Altered Data Box Plot")
    axes[1].set_xlabel("Gender")
    axes[1].set_ylabel("Height (cm)")
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()
    
    plt.tight_layout()
    plt.show()
    
    # --- Part (d): Outlier Detection in Altered Female Data ---
    # Visual Methods.
    plt.figure()
    plt.hist(female_data_altered, bins=30, color='purple', edgecolor='black', alpha=0.7)
    plt.title("Visual Outlier Detection: Histogram of Altered Female Heights")
    plt.xlabel("Height (cm)")
    plt.ylabel("Frequency")
    plt.show()
    
    plt.figure()
    plt.boxplot(female_data_altered, patch_artist=True, boxprops=dict(facecolor='purple'))
    plt.title("Visual Outlier Detection: Boxplot of Altered Female Heights")
    plt.xlabel("Altered Female Data")
    plt.show()
    
    # Parametric and Non-Parametric Methods.
    print("\n=== Parametric Outlier Detection (z-score) ===")
    for cutoff in [2, 3]:
        outliers_z = detect_outliers_zscore(female_data_altered, cutoff)
        print(f"Z-score cutoff {cutoff}: {len(outliers_z)} outliers detected")
    
    print("\n=== Non-Parametric Outlier Detection (IQR) ===")
    for factor in [1.5, 2, 3]:
        outliers_iqr = detect_outliers_iqr(female_data_altered, factor)
        print(f"IQR factor {factor}: {len(outliers_iqr)} outliers detected")
    
    print("\n=== Non-Parametric Outlier Detection (MAD) ===")
    for cutoff in [1.5, 2, 3]:
        outliers_mad = detect_outliers_mad(female_data_altered, cutoff)
        print(f"MAD cutoff {cutoff}: {len(outliers_mad)} outliers detected")
    
    # --- Part (e): Remove Outliers ---
    outlier_indices = detect_outliers_zscore(female_data_altered, cutoff=3)
    female_data_clean = np.delete(female_data_altered, outlier_indices)
    print("\nAfter removing outliers (z-score cutoff = 3):")
    print("Clean Female Data: Mean = {:.2f}, SD = {:.2f}".format(np.mean(female_data_clean), np.std(female_data_clean)))
    plt.figure()
    plt.hist(female_data_clean, bins=30, color='purple', edgecolor='black', alpha=0.7)
    plt.title("Histogram of Cleaned Female Heights (Outliers Removed)")
    plt.xlabel("Height (cm)")
    plt.ylabel("Frequency")
    plt.show()
    plt.figure()
    plt.boxplot(female_data_clean, patch_artist=True, boxprops=dict(facecolor='purple'))
    plt.title("Boxplot of Cleaned Female Heights (Outliers Removed)")
    plt.xlabel("Cleaned Female Data")
    plt.show()
    
    # --- Part (c) & (f): Run Classification on Altered and Cleaned Data ---
    # On Altered Data.
    threshold_val = (np.mean(female_data_altered) + np.mean(male_data)) / 2
    actual_thr, predictions_thr = threshold_classifier(female_data_altered, male_data, threshold_val)
    _, acc_thr = evaluate_classifier(actual_thr, predictions_thr)
    actual_prob, predictions_prob = probability_classifier(female_data_altered, male_data,
                                                           np.mean(female_data_altered), np.std(female_data_altered),
                                                           np.mean(male_data), np.std(male_data))
    _, acc_prob = evaluate_classifier(actual_prob, predictions_prob)
    actual_quant, predictions_quant = quantized_classifier(female_data_altered, male_data, interval_len=1)
    _, acc_quant = evaluate_classifier(actual_quant, predictions_quant)
    print("\nClassification Results on Altered Data:")
    print("Threshold Classifier Accuracy: {:.2f}%".format(acc_thr * 100))
    print("Probability Classifier Accuracy: {:.2f}%".format(acc_prob * 100))
    print("Quantized Classifier Accuracy: {:.2f}%".format(acc_quant * 100))
    
    # On Cleaned Data.
    threshold_clean = (np.mean(female_data_clean) + np.mean(male_data)) / 2
    actual_thr_clean, predictions_thr_clean = threshold_classifier(female_data_clean, male_data, threshold_clean)
    _, acc_thr_clean = evaluate_classifier(actual_thr_clean, predictions_thr_clean)
    actual_prob_clean, predictions_prob_clean = probability_classifier(female_data_clean, male_data,
                                                                      np.mean(female_data_clean), np.std(female_data_clean),
                                                                      np.mean(male_data), np.std(male_data))
    _, acc_prob_clean = evaluate_classifier(actual_prob_clean, predictions_prob_clean)
    actual_quant_clean, predictions_quant_clean = quantized_classifier(female_data_clean, male_data, interval_len=1)
    _, acc_quant_clean = evaluate_classifier(actual_quant_clean, predictions_quant_clean)
    print("\nClassification Results on Cleaned Data (Outliers Removed):")
    print("Threshold Classifier Accuracy: {:.2f}%".format(acc_thr_clean * 100))
    print("Probability Classifier Accuracy: {:.2f}%".format(acc_prob_clean * 100))
    print("Quantized Classifier Accuracy: {:.2f}%".format(acc_quant_clean * 100))
    
    # --- Part (g): Data Trimming Experiment ---
    # Create a combined DataFrame from the altered female data and male data.
    df_altered_combined = pd.DataFrame({
        'height': np.concatenate([female_data_altered, male_data]),
        'label': np.concatenate([['F'] * len(female_data_altered), ['M'] * len(male_data)])
    })
    df_sorted = df_altered_combined.sort_values(by='height').reset_index(drop=True)
    n_total = len(df_sorted)
    k_values = list(range(1, 16))
    acc_thr_list, acc_prob_list, acc_quant_list = [], [], []
    
    for k in k_values:
        k_lower = int(n_total * k / 100)
        k_upper = int(n_total * (1 - k / 100))
        df_trimmed = df_sorted.iloc[k_lower:k_upper].copy()
        trimmed_female = df_trimmed[df_trimmed['label'] == 'F']['height'].values
        trimmed_male = df_trimmed[df_trimmed['label'] == 'M']['height'].values
        if len(trimmed_female) == 0 or len(trimmed_male) == 0:
            continue
        threshold_trim = (np.mean(trimmed_female) + np.mean(trimmed_male)) / 2
        actual_thr_trim, predictions_thr_trim = threshold_classifier(trimmed_female, trimmed_male, threshold_trim)
        _, acc_thr_trim = evaluate_classifier(actual_thr_trim, predictions_thr_trim)
        acc_thr_list.append(acc_thr_trim)
        actual_prob_trim, predictions_prob_trim = probability_classifier(trimmed_female, trimmed_male,
                                                                        np.mean(trimmed_female), np.std(trimmed_female),
                                                                        np.mean(trimmed_male), np.std(trimmed_male))
        _, acc_prob_trim = evaluate_classifier(actual_prob_trim, predictions_prob_trim)
        acc_prob_list.append(acc_prob_trim)
        actual_quant_trim, predictions_quant_trim = quantized_classifier(trimmed_female, trimmed_male, interval_len=1)
        _, acc_quant_trim = evaluate_classifier(actual_quant_trim, predictions_quant_trim)
        acc_quant_list.append(acc_quant_trim)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values[:len(acc_thr_list)], [acc * 100 for acc in acc_thr_list],
             marker='o', linestyle='-', color='red', label='Threshold Classifier')
    plt.plot(k_values[:len(acc_prob_list)], [acc * 100 for acc in acc_prob_list],
             marker='s', linestyle='-', color='blue', label='Probability Classifier')
    plt.plot(k_values[:len(acc_quant_list)], [acc * 100 for acc in acc_quant_list],
             marker='^', linestyle='-', color='orange', label='Quantized Classifier')
    plt.xlabel("Trimming Percentage (k%)")
    plt.ylabel("Classification Accuracy (%)")
    plt.title("Impact of Data Trimming on Classification Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main_assignment3()
        