import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, accuracy_score

# Function to generate height distributions
def generate_heights(mean, std_dev, size, label):
    heights = np.random.normal(mean, std_dev, size)
    labels = [label] * size
    return pd.DataFrame({'height': heights, 'label': labels})

# Function to plot histograms
def plot_histograms(female_heights, male_heights, bins=50):
    plt.hist([female_heights, male_heights], bins=bins, label=['Female', 'Male'], alpha=0.7, color=['purple', 'green'])
    plt.title('Height Distributions')
    plt.xlabel('Height (cm)')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

# Function for threshold classification
def threshold_classifier(female_data, male_data, threshold):
    predictions = np.where(np.concatenate([female_data, male_data]) < threshold, 'F', 'M')
    actual = np.concatenate([['F'] * len(female_data), ['M'] * len(male_data)])
    return actual, predictions

# Function for probability-based classification
def probability_classifier(female_data, male_data, female_mean, female_sd, male_mean, male_sd):
    def classify(height):
        female_prob = norm.pdf(height, female_mean, female_sd)
        male_prob = norm.pdf(height, male_mean, male_sd)
        return 'F' if female_prob > male_prob else 'M'
    
    predictions = np.array([classify(h) for h in np.concatenate([female_data, male_data])])
    actual = np.concatenate([['F'] * len(female_data), ['M'] * len(male_data)])
    return actual, predictions

# Function for quantized classification
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

# Function to evaluate and print confusion matrix and accuracy
def evaluate_classifier(actual, predictions):
    cm = confusion_matrix(actual, predictions, labels=['F', 'M'])
    accuracy = accuracy_score(actual, predictions)
    print("Confusion Matrix:\n", cm)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    return cm, accuracy

# Main function to execute experiments
def main():
    # Parameters
    female_mean = 152
    male_mean = 166
    std_devs = [2.5, 5, 7.5, 10]
    quantization_intervals = [0.001, 0.05, 0.1, 0.3, 1, 2, 5, 10]
    sample_size = 1000
    
    for sd in std_devs:
        print(f"\n--- Standard Deviation: {sd} ---")
        
        # Generate data
        female_data = generate_heights(female_mean, sd, sample_size, 'F')['height'].values
        male_data = generate_heights(male_mean, sd, sample_size, 'M')['height'].values
        
        # Plot histograms
        plot_histograms(female_data, male_data)
        
        # Threshold-based classification
        threshold = (female_mean + male_mean) / 2
        print(f"\nThreshold Classifier (Threshold = {threshold}):")
        actual, predictions = threshold_classifier(female_data, male_data, threshold)
        evaluate_classifier(actual, predictions)
        
        # Probability-based classification
        print("\nProbability Classifier:")
        actual, predictions = probability_classifier(female_data, male_data, female_mean, sd, male_mean, sd)
        evaluate_classifier(actual, predictions)
        
        # Quantized classification
        for interval_len in quantization_intervals:
            print(f"\nQuantized Classifier (Interval Length = {interval_len}):")
            actual, predictions = quantized_classifier(female_data, male_data, interval_len)
            evaluate_classifier(actual, predictions)

# Run the main function
if __name__ == "__main__":
    main()
