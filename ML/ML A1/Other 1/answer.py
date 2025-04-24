import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate height data
f_heights = pd.Series(np.random.normal(152, 5, 1000))
m_heights = pd.Series(np.random.normal(166, 5, 1000))

# Plot histograms
plt.hist([f_heights, m_heights], bins=100, label=['female', 'male'])
plt.legend(loc='upper right')
plt.show()

# Threshold Classifier
def threshold_classifier(threshold_increment, f_heights, m_heights):
    lower_bound_of_overlap = m_heights.min()
    upper_bound_of_overlap = f_heights.max()
    total = f_heights.size + m_heights.size

    new_lower_bound = np.floor(lower_bound_of_overlap)
    new_upper_bound = np.ceil(upper_bound_of_overlap)

    current_min_miss_classification_rate = 100.0
    current_optimal_threshold = new_lower_bound

    for threshold in np.arange(new_lower_bound, new_upper_bound + 1, threshold_increment):
        misclassified_females = sum(f_heights >= threshold)
        misclassified_males = sum(m_heights < threshold)
        misclassification_rate = 100.0 * (misclassified_females + misclassified_males) / total

        if misclassification_rate < current_min_miss_classification_rate:
            current_min_miss_classification_rate = misclassification_rate
            current_optimal_threshold = threshold

    return current_optimal_threshold, current_min_miss_classification_rate

# Probability Classifier
def probability_classifier(f_heights, m_heights):
    male_mean = m_heights.mean()
    male_sd = m_heights.std()
    female_mean = f_heights.mean()
    female_sd = f_heights.std()

    num_misclassified_females = 0
    num_misclassified_males = 0

    # Check misclassified females
    for current_height in f_heights:
        female_probability = norm.pdf(current_height, female_mean, female_sd)
        male_probability = norm.pdf(current_height, male_mean, male_sd)
        if male_probability > female_probability:
            num_misclassified_females += 1

    # Check misclassified males
    for current_height in m_heights:
        female_probability = norm.pdf(current_height, female_mean, female_sd)
        male_probability = norm.pdf(current_height, male_mean, male_sd)
        if male_probability < female_probability:
            num_misclassified_males += 1

    total = f_heights.size + m_heights.size
    misclassification_rate = 100.0 * (num_misclassified_females + num_misclassified_males) / total
    return misclassification_rate, num_misclassified_females, num_misclassified_males

# Quantized Classifier
def quantize(heights, interval_len):
    interval_label = np.floor(heights / interval_len)
    interval_counts = interval_label.value_counts()
    return interval_counts

def local_classifier(interval_len, f_heights, m_heights):
    total = f_heights.size + m_heights.size
    male_quantized = quantize(m_heights, interval_len)
    female_quantized = quantize(f_heights, interval_len)

    quantized_overlap_lower_bound = int(male_quantized.index.min())
    quantized_overlap_upper_bound = int(female_quantized.index.max())

    total_misclassification = 0

    for common_interval in range(quantized_overlap_lower_bound, quantized_overlap_upper_bound + 1):
        female_count = female_quantized.get(common_interval, 0)
        male_count = male_quantized.get(common_interval, 0)

        error = min(female_count, male_count)
        total_misclassification += error

    return 100.0 * total_misclassification / total

# Main Execution
# Threshold Classifier Results
optimal_threshold, min_misclassification_rate = threshold_classifier(0.5, f_heights, m_heights)
print(f"Optimal Threshold: {optimal_threshold}")
print(f"Min Misclassification Rate: {min_misclassification_rate:.2f}%")

# Probability Classifier Results
probability_misclassification_rate, num_females_misclassified, num_males_misclassified = probability_classifier(f_heights, m_heights)
print(f"Probability Misclassification Rate: {probability_misclassification_rate:.2f}%")
print(f"Number of Misclassified Females: {num_females_misclassified}")
print(f"Number of Misclassified Males: {num_males_misclassified}")

# Local Classifier Results
local_error = local_classifier(0.5, f_heights, m_heights)
print(f"Local Classifier Error: {local_error:.2f}%")
