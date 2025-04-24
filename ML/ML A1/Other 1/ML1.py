import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

f_heights = pd.Series(np.random.normal(152, 5, 1000))
m_heights = pd.Series(np.random.normal(166, 5, 1000))

num_females = f_heights.size
num_males = m_heights.size
total = num_females + num_males

plt.hist([f_heights, m_heights], bins=100, label=['female', 'male'])
plt.legend(loc='upper right')
plt.show()

def threshold_classifier(threshold_increment, f_heights, m_heights):
    lower_bound_of_overlap = m_heights.min()
    upper_bound_of_overlap = f_heights.max()
    total = f_heights.size + m_heights.size

    print(lower_bound_of_overlap, upper_bound_of_overlap)

    new_lower_bound = np.floor(lower_bound_of_overlap)
    new_upper_bound = np.ceil(upper_bound_of_overlap)

    current_min_miss_classification_rate = 100.0
    current_optimal_threshold = new_lower_bound

    for threshold in np.arange(new_lower_bound, new_upper_bound+1, threshold_increment):
        misclassified_females = sum(f_heights >= threshold)
        misclassified_males = sum(m_heights < threshold)
        misclassification_rate = 100.0 * (misclassified_females + misclassified_males) / total

        print(threshold, misclassified_females, misclassified_males, misclassification_rate)

        if misclassification_rate < current_min_miss_classification_rate:
            current_min_miss_classification_rate = misclassification_rate
            current_optimal_threshold = threshold

    print("currentmin", current_optimal_threshold)
    return current_optimal_threshold, current_min_miss_classification_rate

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
        female_count = 0 if common_interval not in female_quantized.index else female_quantized[common_interval]
        male_count = 0 if common_interval not in male_quantized.index else male_quantized[common_interval]

        error = min(female_count, male_count)
        total_misclassification += error

        print(common_interval, female_count, male_count, error, total_misclassification)

    return 100.0 * total_misclassification / total

def probability_classifier(f_heights, m_heights):
    total = f_heights.size + m_heights.size
    male_mean = m_heights.mean()
    male_sd= m_heights.std()
    female_mean = f_heights.mean()
    female_sd= f_heights.std()
    print(male_mean, male_sd, female_mean, female_sd)

    num_misclassified_females = 0
    for current_height in f_heights:
        female_probability = norm.pdf(current_height,female_mean,female_sd)
        male_probability = norm.pdf(current_height,male_mean,male_sd)
        if male_probability > female_probability:
            print("misclassified female", current_height,female_probability, male_probability)
            num_misclassified_females +=1

    print("num_misclassified_females =", num_misclassified_females)

    num_misclassified_males = 0
    for current_height in m_heights:
        female_probability = norm.pdf(current_height,female_mean,female_sd)
        male_probability = norm.pdf(current_height,male_mean,male_sd)
        if male_probability < female_probability:
            print("misclassified male",current_height,female_probability, male_probability)
            num_misclassified_males +=1

    print("num_misclassified_males =", num_misclassified_males)
    return num_misclassified_females, num_misclassified_males

threshold_results = threshold_classifier(0.5, f_heights, m_heights)
print(threshold_results)

local_error = local_classifier(0.5, f_heights, m_heights)
print(f'Local Classifier Error: {local_error}')

probability_results = probability_classifier(f_heights, m_heights)
print(f'Probability Classifier Results: {probability_results}')
