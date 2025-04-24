import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, accuracy_score

# Generate height data for males and females
def gen_data(mean, sd, size, label):
    heights = np.random.normal(mean, sd, size)
    return pd.DataFrame({'height': heights, 'label': [label] * size})

# Updated plotting function to match code1's style
def plot_hist(female, male, bins=50, label=""):
    plt.hist([female, male], bins=bins, label=['Female', 'Male'], alpha=0.7, color=['purple', 'green'])
    plt.title(f'Height Distributions {label}')
    plt.xlabel('Height (cm)')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

# Threshold-based classification
def threshold_class(female, male, threshold):
    female_preds = np.where(female < threshold, 'F', 'M')
    male_preds = np.where(male < threshold, 'F', 'M')
    female_labels = np.full(len(female), 'F')
    male_labels = np.full(len(male), 'M')
    return np.append(female_labels, male_labels), np.append(female_preds, male_preds)

# Probability-based classification
def prob_class(female, male, f_mean, f_sd, m_mean, m_sd):
    def classify(h, mean, sd, alt_mean, alt_sd):
        f_prob = norm.pdf(h, mean, sd)
        m_prob = norm.pdf(h, alt_mean, alt_sd)
        return 'F' if f_prob > m_prob else 'M'

    female_preds = np.array([classify(h, f_mean, f_sd, m_mean, m_sd) for h in female])
    male_preds = np.array([classify(h, m_mean, m_sd, f_mean, f_sd) for h in male])
    female_labels = np.full(len(female), 'F')
    male_labels = np.full(len(male), 'M')
    return np.append(female_labels, male_labels), np.append(female_preds, male_preds)

# Quantized classification
def quant_class(female, male, interval):
    def quantize(data):
        bins = np.floor(data / interval).astype(int)
        bin_counts = {}
        for b in bins:
            bin_counts[b] = bin_counts.get(b, 0) + 1
        return bin_counts

    female_counts = quantize(female)
    male_counts = quantize(male)

    # Classify bins based on counts
    female_labels, female_preds = [], []
    for b, count in female_counts.items():
        male_count = male_counts.get(b, 0)
        pred_label = 'F' if count >= male_count else 'M'
        female_labels.extend(['F'] * count)
        female_preds.extend([pred_label] * count)

    male_labels, male_preds = [], []
    for b, count in male_counts.items():
        female_count = female_counts.get(b, 0)
        pred_label = 'F' if female_count >= count else 'M'
        male_labels.extend(['M'] * count)
        male_preds.extend([pred_label] * count)

    return np.array(female_labels + male_labels), np.array(female_preds + male_preds)

# Evaluate classifier performance
def evaluate(true_labels, preds, label=""):
    cm = confusion_matrix(true_labels, preds, labels=['F', 'M'])
    accuracy = accuracy_score(true_labels, preds)
    print(f"\n{label}")
    print("Confusion Matrix:")
    print(f"\n\t\tPredicted F\tPredicted M\nActual F\t{cm[0, 0]:<5} (TP)\t{cm[0, 1]:<5} (FN)\nActual M\t{cm[1, 0]:<5} (FP)\t{cm[1, 1]:<5} (TN)")
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    return cm, accuracy

# Main function to run experimentation
def run():
    f_mean = 152
    m_mean = 166
    std_devs = [2.5, 5, 7.5, 10]
    intervals = [0.001, 0.05, 0.1, 0.3, 1, 2, 5, 10]
    size = 1000
    results = []
    
    for sd in std_devs:
        print(f"\n--- SD = {sd} ---")
        
        # Generate synthetic data
        female = gen_data(f_mean, sd, size, 'F')['height'].values
        male = gen_data(m_mean, sd, size, 'M')['height'].values
        
        # Plot histograms using the updated plot_hist function
        plot_hist(female, male, label=f'(SD={sd})')
        
        # Threshold classification
        threshold = (f_mean + m_mean) / 2
        true_labels, preds = threshold_class(female, male, threshold)
        cm, acc = evaluate(true_labels, preds, label=f"Threshold (Threshold={threshold})")
        results.append({"Method": "Threshold", "SD": sd, "Interval": None, "Accuracy": acc})
        
        # Probability classification
        true_labels, preds = prob_class(female, male, f_mean, sd, m_mean, sd)
        cm, acc = evaluate(true_labels, preds, label="Probability Classifier")
        results.append({"Method": "Probability", "SD": sd, "Interval": None, "Accuracy": acc})
        
        # Quantized classification
        for interval in intervals:
            true_labels, preds = quant_class(female, male, interval)
            cm, acc = evaluate(true_labels, preds, label=f"Quantized (Interval={interval})")
            results.append({"Method": "Quantized", "SD": sd, "Interval": interval, "Accuracy": acc})
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('myresult.csv', index=False)
    print("\nResults saved to 'results.csv'.")

# Run the experiments
run()
