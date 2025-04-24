import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, accuracy_score

# Generate height data for males and females
def gen_data(mean, sd, size, label):
    heights = np.random.normal(mean, sd, size)
    return pd.DataFrame({'height': heights, 'label': [label] * size})

# Plotting the height dist. for both males and females
def plot_hist(female, male, bins=50, label=""):
    plt.hist([female, male], bins=bins, label=['Female', 'Male'], alpha=0.7, color=['purple', 'green'])
    plt.title(f'Height Distribution {label}')
    plt.xlabel('Height (cm)')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

# Threshold-based classification
def threshold_class(female, male, threshold):
    data = np.concatenate([female, male])
    preds = np.where(data < threshold, 'F', 'M')
    true_labels = np.concatenate([['F'] * len(female), ['M'] * len(male)])
    return true_labels, preds

# Prob-based classification
def prob_class(female, male, f_mean, f_sd, m_mean, m_sd):
    def classify(h):
        f_prob = norm.pdf(h, f_mean, f_sd)
        m_prob = norm.pdf(h, m_mean, m_sd)
        return 'F' if f_prob > m_prob else 'M'
    
    data = np.concatenate([female, male])
    preds = np.array([classify(h) for h in data])
    true_labels = np.concatenate([['F'] * len(female), ['M'] * len(male)])
    return true_labels, preds

# Quantized classification
def quant_class(female, male, interval):
    def quantize(data):
        return pd.Series(np.floor(data / interval)).value_counts()
    
    f_quant = quantize(female)
    m_quant = quantize(male)
    intervals = set(f_quant.index).union(set(m_quant.index))
    
    preds = []
    true_labels = []
    
    for interval in intervals:
        f_count = f_quant.get(interval, 0)
        m_count = m_quant.get(interval, 0)
        label = 'F' if f_count >= m_count else 'M'
        preds.extend([label] * (f_count + m_count))
        true_labels.extend(['F'] * f_count + ['M'] * m_count)
    
    return np.array(true_labels), np.array(preds)

# Evaluate classifier performance
def evaluate(true_labels, preds, label=""):
    cm = confusion_matrix(true_labels, preds, labels=['F', 'M'])
    accuracy = accuracy_score(true_labels, preds)
    print(f"\n{label}")
    print("Confusion Matrix:")
    print(f"\n\t\tPredicted F\tPredicted M\nActual F\t{cm[0, 0]:<5} (TP)\t{cm[0, 1]:<5} (FN)\nActual M\t{cm[1, 0]:<5} (FP)\t{cm[1, 1]:<5} (TN)")
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    return cm, accuracy

# Main function to run other values for experimentation
def run():
    f_mean = 152
    m_mean = 166
    std_devs = [2.5, 5, 7.5, 10]
    intervals = [0.001, 0.05, 0.1, 0.3, 1, 2, 5, 10]
    size = 1000
    results = []
    
    for sd in std_devs:
        print(f"\n--- SD = {sd} ---")
        
        # Generation of synthetic data
        female = gen_data(f_mean, sd, size, 'F')['height'].values
        male = gen_data(m_mean, sd, size, 'M')['height'].values
        
        # Plotting histograms
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
    df.to_csv('results.csv', index=False)
    print("\nResults saved to 'results.csv'.")

run()
