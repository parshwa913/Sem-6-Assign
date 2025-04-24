import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# For reproducibility
np.random.seed(42)

# Parameters
n_samples = 1000
female_mean, male_mean = 152, 166
sd = 5

# Generate samples
female_heights = np.random.normal(loc=female_mean, scale=sd, size=n_samples)
male_heights = np.random.normal(loc=male_mean, scale=sd, size=n_samples)

# Create DataFrames with labels
df_female = pd.DataFrame({'height': female_heights, 'gender': 'F'})
df_male = pd.DataFrame({'height': male_heights, 'gender': 'M'})

# Combine into one dataset
df = pd.concat([df_female, df_male], ignore_index=True)

# Plotting histogram and box plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.histplot(data=df, x="height", hue="gender", bins=100, stat="density", common_norm=False, ax=axes[0])
axes[0].set_title("Histogram of Heights by Gender")
axes[0].set_xlabel("Height (cm)")
axes[0].set_ylabel("Density")

sns.boxplot(data=df, x="gender", y="height", ax=axes[1])
axes[1].set_title("Box Plot of Heights by Gender")
axes[1].set_xlabel("Gender")
axes[1].set_ylabel("Height (cm)")

plt.tight_layout()
plt.show()

# Sort the female data by height in descending order and select the top 50 indices
female_data_sorted = df_female.sort_values(by='height', ascending=False)
top_50_indices = female_data_sorted.head(50).index

# Create a new (altered) dataset by copying the original
df_altered = df.copy()

# Increase the top 50 female height values by 10 cm each
df_altered.loc[top_50_indices, 'height'] += 10

# Plotting histogram and box plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.histplot(data=df_altered, x="height", hue="gender", bins=100, stat="density", common_norm=False, ax=axes[0])
axes[0].set_title("Histogram of Altered Heights by Gender")
axes[0].set_xlabel("Height (cm)")
axes[0].set_ylabel("Density")

sns.boxplot(data=df_altered, x="gender", y="height", ax=axes[1])
axes[1].set_title("Box Plot of Altered Heights by Gender")
axes[1].set_xlabel("Gender")
axes[1].set_ylabel("Height (cm)")

plt.tight_layout()
plt.show()

# Statistics for female heights before alteration
orig_female_stats = df[df['gender'] == 'F']['height'].describe()

# Statistics for female heights after alteration
altered_female_stats = df_altered[df_altered['gender'] == 'F']['height'].describe()

print("Original Female Height Statistics:")
print(orig_female_stats)
print("\nAltered Female Height Statistics (after adding 10 cm to top 50):")
print(altered_female_stats)

def gaussian_pdf(x, mean, sd):
    return (1/(sd * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mean)/sd)**2)

# --- 1. Likelihood‐based Classifier ---
def likelihood_classifier(df_input, gender_col='gender', height_col='height'):
    params = df_input.groupby(gender_col)[height_col].agg(['mean', 'std']).to_dict('index')
    predictions = []
    for h in df_input[height_col]:
        likelihood_F = gaussian_pdf(h, params['F']['mean'], params['F']['std'])
        likelihood_M = gaussian_pdf(h, params['M']['mean'], params['M']['std'])
        predictions.append('F' if likelihood_F > likelihood_M else 'M')
    return predictions

# --- 2. Threshold‐based Classifier ---
def threshold_classifier(df_input, gender_col='gender', height_col='height'):
    mean_F = df_input[df_input[gender_col]=='F'][height_col].mean()
    mean_M = df_input[df_input[gender_col]=='M'][height_col].mean()
    threshold = (mean_F + mean_M) / 2
    predictions = df_input[height_col].apply(lambda x: 'F' if x < threshold else 'M')
    return predictions, threshold

# --- 3. Quantization‐based Classifier ---
def quantization_classifier(df_input, bin_width=0.5, gender_col='gender', height_col='height'):
    min_height = df_input[height_col].min()
    max_height = df_input[height_col].max()
    bins = np.arange(min_height, max_height + bin_width, bin_width)
    
    df_temp = df_input.copy()
    df_temp['bin'] = pd.cut(df_temp[height_col], bins=bins, include_lowest=True)
    
    bin_majority = df_temp.groupby('bin', observed=False)[gender_col].agg(
        lambda x: x.value_counts().idxmax() if not x.empty else None
    )
    
    predictions = df_temp['bin'].map(bin_majority).fillna('M')
    return predictions

    # Apply classifiers on the altered dataset
res_likelihood = likelihood_classifier(df_altered)
res_threshold, threshold_value = threshold_classifier(df_altered)
res_quantization = quantization_classifier(df_altered)

# Compute confusion matrices and accuracies
true_labels = df_altered['gender']

cm_likelihood = confusion_matrix(true_labels, res_likelihood, labels=['F','M'])
acc_likelihood = accuracy_score(true_labels, res_likelihood)

cm_threshold = confusion_matrix(true_labels, res_threshold, labels=['F','M'])
acc_threshold = accuracy_score(true_labels, res_threshold)

cm_quantization = confusion_matrix(true_labels, res_quantization, labels=['F','M'])
acc_quantization = accuracy_score(true_labels, res_quantization)

print("Likelihood-based Classifier")
print("Confusion Matrix:\n", cm_likelihood)
print("Accuracy: {:.2f}".format(acc_likelihood))

print("\nThreshold-based Classifier (Threshold = {:.2f})".format(threshold_value))
print("Confusion Matrix:\n", cm_threshold)
print("Accuracy: {:.2f}".format(acc_threshold))

print("\nQuantization-based Classifier")
print("Confusion Matrix:\n", cm_quantization)
print("Accuracy: {:.2f}".format(acc_quantization))

# Plotting histogram and box plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.histplot(df_altered[df_altered['gender']=='F']['height'], bins=30, stat="density", common_norm=False, ax=axes[0])
axes[0].set_title("Histogram of Altered Female Heights")
axes[0].set_xlabel("Height (cm)")
axes[0].set_ylabel("Frequency")

sns.boxplot(x=df_altered[df_altered['gender']=='F']['height'], ax=axes[1])
axes[1].set_title("Boxplot of Altered Female Heights")
axes[1].set_xlabel("Height (cm)")

plt.tight_layout()
plt.show()

# Extract altered female heights
female_altered = df_altered[df_altered['gender']=='F']['height']

# Compute mean and std for the altered female sample
female_mean_altered = female_altered.mean()
female_std_altered = female_altered.std()

# Compute z-scores
z_scores = (female_altered - female_mean_altered) / female_std_altered

# Identify outliers using cutoffs
outliers_z2 = female_altered[np.abs(z_scores) > 2]
outliers_z3 = female_altered[np.abs(z_scores) > 3]

print("Number of outliers with z > 2:", len(outliers_z2))
print("Number of outliers with z > 3:", len(outliers_z3))

# IQR Method for altered female heights
Q1 = female_altered.quantile(0.25)
Q3 = female_altered.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = female_altered[(female_altered < lower_bound) | (female_altered > upper_bound)]
print("Number of outliers detected by IQR method:", len(outliers_iqr))

# MAD Method for altered female heights
median_female = female_altered.median()
MAD = np.median(np.abs(female_altered - median_female))

# Compute modified z-scores
# 0.6745 is the constant, MAD is approximately 0.6745 * std
modified_z_score = 0.6745 * (female_altered - median_female) / MAD
cutoff = 3
outliers_mad = female_altered[np.abs(modified_z_score) > cutoff]
print("Number of outliers detected by MAD method (cutoff {}):".format(cutoff), len(outliers_mad))

# Remove outliers from female samples (using IQR bounds)
df_cleaned = df_altered.copy()

# Boolean mask for female rows
female_mask = df_cleaned['gender'] == 'F'

# Remove lower and upper outliers for females using .loc[]
df_cleaned = df_cleaned.loc[~((female_mask) & (df_cleaned['height'] < lower_bound))]
df_cleaned = df_cleaned.loc[~((female_mask) & (df_cleaned['height'] > upper_bound))]

print("Shape before cleaning:", df_altered.shape)
print("Shape after cleaning:", df_cleaned.shape)
print("Number of outliers removed:", df_altered.shape[0] - df_cleaned.shape[0])

print("Original female height statistics:")
print(df[df['gender'] == 'F']['height'].describe())

print("\nFemale height statistics after outlier removal:")
print(df_cleaned[df_cleaned['gender'] == 'F']['height'].describe())

# Apply classifiers on the cleaned dataset
res_likelihood_cleaned = likelihood_classifier(df_cleaned)
res_threshold_cleaned, threshold_value_cleaned = threshold_classifier(df_cleaned)
res_quantization_cleaned = quantization_classifier(df_cleaned)

# Compute confusion matrices and accuracies for the cleaned dataset
true_labels_cleaned = df_cleaned['gender']

cm_likelihood_cleaned = confusion_matrix(true_labels_cleaned, res_likelihood_cleaned, labels=['F','M'])
acc_likelihood_cleaned = accuracy_score(true_labels_cleaned, res_likelihood_cleaned)

cm_threshold_cleaned = confusion_matrix(true_labels_cleaned, res_threshold_cleaned, labels=['F','M'])
acc_threshold_cleaned = accuracy_score(true_labels_cleaned, res_threshold_cleaned)

cm_quantization_cleaned = confusion_matrix(true_labels_cleaned, res_quantization_cleaned, labels=['F','M'])
acc_quantization_cleaned = accuracy_score(true_labels_cleaned, res_quantization_cleaned)

print("Likelihood-based Classifier (After Outlier Removal)")
print("Confusion Matrix:\n", cm_likelihood_cleaned)
print("Accuracy: {:.2f}".format(acc_likelihood_cleaned))

print("\nThreshold-based Classifier (Threshold = {:.2f}) (After Outlier Removal)".format(threshold_value_cleaned))
print("Confusion Matrix:\n", cm_threshold_cleaned)
print("Accuracy: {:.2f}".format(acc_threshold_cleaned))

print("\nQuantization-based Classifier (After Outlier Removal)")
print("Confusion Matrix:\n", cm_quantization_cleaned)
print("Accuracy: {:.2f}".format(acc_quantization_cleaned))

# Define the range of trimming percentages
k_values = np.arange(1, 16)
accuracy_likelihood = []
accuracy_threshold = []
accuracy_quantization = []

# Sort the overall dataset by height for trimming
df_sorted = df.sort_values(by='height').reset_index(drop=True)
n_total = len(df_sorted)

for k in k_values:
    k_lower = int(n_total * k/100)
    k_upper = int(n_total * (1 - k/100))
    df_trimmed = df_sorted.iloc[k_lower:k_upper].copy()
    
    # Apply classifiers on the trimmed dataset
    res_likelihood_trim = likelihood_classifier(df_trimmed)
    res_threshold_trim, _ = threshold_classifier(df_trimmed)
    res_quantization_trim = quantization_classifier(df_trimmed)
    
    true_labels_trim = df_trimmed['gender']
    
    accuracy_likelihood.append(accuracy_score(true_labels_trim, res_likelihood_trim))
    accuracy_threshold.append(accuracy_score(true_labels_trim, res_threshold_trim))
    accuracy_quantization.append(accuracy_score(true_labels_trim, res_quantization_trim))

# Plot the impact of trimming on classification accuracy
plt.figure(figsize=(10,6))
plt.plot(k_values, accuracy_likelihood, marker='o', label='Likelihood-based')
plt.plot(k_values, accuracy_threshold, marker='s', label='Threshold-based')
plt.plot(k_values, accuracy_quantization, marker='^', label='Quantization-based')
plt.xlabel("Trimming Percentage k% (Dropped from each end)")
plt.ylabel("Classification Accuracy")
plt.title("Impact of Data Trimming on Classification Accuracy")
plt.legend()
plt.grid(True)
plt.show()

