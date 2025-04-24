import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def generate_data():
    np.random.seed(42)
    male_heights = np.random.normal(166, 5.5, 1000)
    female_heights = np.random.normal(152, 4.5, 1000)
    male_labels = np.zeros(1000)
    female_labels = np.ones(1000)
    male_idx = np.random.permutation(1000)
    female_idx = np.random.permutation(1000)
    male_test = male_heights[male_idx[:200]]
    male_train = male_heights[male_idx[200:]]
    female_test = female_heights[female_idx[:200]]
    female_train = female_heights[female_idx[200:]]
    male_test_labels = male_labels[male_idx[:200]]
    male_train_labels = male_labels[male_idx[200:]]
    female_test_labels = female_labels[female_idx[:200]]
    female_train_labels = female_labels[female_idx[200:]]
    X_train = np.array(list(male_train) + list(female_train)).reshape(-1, 1)
    y_train = np.array(list(male_train_labels) + list(female_train_labels))
    X_test = np.array(list(male_test) + list(female_test)).reshape(-1, 1)
    y_test = np.array(list(male_test_labels) + list(female_test_labels))
    return male_train, female_train, male_train_labels, female_train_labels, X_train, y_train, X_test, y_test

def train_classifier(X_train, y_train, X_test, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    return train_acc, test_acc, clf

def plot_histogram(data, title, color):
    plt.hist(data, bins=20, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel("Height (cm)")
    plt.ylabel("Frequency")

def plot_boxplot(data, title, color):
    plt.boxplot(data, vert=True, patch_artist=True, boxprops={'facecolor': color, 'color': 'black'}, medianprops={'color': 'red'})
    plt.title(title)
    plt.ylabel("Height (cm)")

def inject_outliers(data):
    data_mod = data.copy()
    top50_indices = np.argsort(data_mod)[-50:]
    data_mod[top50_indices] += 10
    return data_mod

def remove_outliers_zscore(data, threshold=2.5):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    mask = np.abs(z_scores) < threshold
    data_clean = data[mask]
    return data_clean, len(data) - len(data_clean)

def trimming_analysis(female_data, male_train, male_train_labels, X_test, y_test):
    trim_percentages = list(range(1, 16))
    train_acc_list = []
    test_acc_list = []
    for k in trim_percentages:
        n_trim = int(len(female_data) * k / 100)
        sorted_female = np.sort(female_data)
        trimmed_female = sorted_female[n_trim: len(sorted_female) - n_trim]
        print("\nFor {}% trimming:".format(k))
        print(" - Number of female samples after trimming: {}".format(len(trimmed_female)))
        print(" - First 10 values: ", np.round(trimmed_female[:10], 2))
        print(" - Last 10 values:  ", np.round(trimmed_female[-10:], 2))
        X_trim = np.array(list(male_train) + list(trimmed_female)).reshape(-1, 1)
        y_trim = np.array(list(male_train_labels) + list(np.ones(len(trimmed_female))))
        train_acc, test_acc, _ = train_classifier(X_trim, y_trim, X_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(" - Train Accuracy: {:.4f}".format(train_acc))
        print(" - Test Accuracy:  {:.4f}".format(test_acc))
    plt.figure(figsize=(8, 5))
    plt.plot(trim_percentages, train_acc_list, marker='o', label='Train Accuracy')
    plt.plot(trim_percentages, test_acc_list, marker='s', label='Test Accuracy')
    plt.xlabel("Trim Percentage (%)")
    plt.ylabel("Accuracy")
    plt.title("Impact of Trimming on Classifier Accuracy")
    plt.xticks(trim_percentages)
    plt.legend()
    plt.grid(True)
    plt.show()

male_train, female_train, male_train_labels, female_train_labels, X_train, y_train, X_test, y_test = generate_data()
train_acc, test_acc, _ = train_classifier(X_train, y_train, X_test, y_test)
print("Initial Classifier:")
print("Train Accuracy: {:.4f}".format(train_acc))
print("Test Accuracy:  {:.4f}\n".format(test_acc))
female_train_mod = inject_outliers(female_train)
mean_before = np.mean(female_train)
std_before = np.std(female_train)
mean_after = np.mean(female_train_mod)
std_after = np.std(female_train_mod)
print("After Outlier Injection (Female Train):")
print("Mean before: {:.2f}, Std before: {:.2f}".format(mean_before, std_before))
print("Mean after:  {:.2f}, Std after:  {:.2f}\n".format(mean_after, std_after))
X_train_mod = np.array(list(male_train) + list(female_train_mod)).reshape(-1, 1)
y_train_mod = np.array(list(male_train_labels) + list(female_train_labels))
train_acc_mod, test_acc_mod, _ = train_classifier(X_train_mod, y_train_mod, X_test, y_test)
print("Classifier After Outlier Injection:")
print("Train Accuracy: {:.4f}".format(train_acc_mod))
print("Test Accuracy:  {:.4f}\n".format(test_acc_mod))
female_train_clean, num_removed = remove_outliers_zscore(female_train_mod, 2.5)
print("Removed {} outliers from female train data using z-score.\n".format(num_removed))
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plot_histogram(female_train, "Original Data Histogram", "purple")
plt.subplot(2, 3, 2)
plot_histogram(female_train_mod, "Injected Data Histogram", "lightcoral")
plt.subplot(2, 3, 3)
plot_histogram(female_train_clean, "Cleaned Data Histogram", "lightgreen")
plt.subplot(2, 3, 4)
plot_boxplot(female_train, "Original Data Box Plot", "purple")
plt.subplot(2, 3, 5)
plot_boxplot(female_train_mod, "Injected Data Box Plot", "lightcoral")
plt.subplot(2, 3, 6)
plot_boxplot(female_train_clean, "Cleaned Data Box Plot", "lightgreen")
plt.tight_layout()
plt.show()
X_train_clean = np.array(list(male_train) + list(female_train_clean)).reshape(-1, 1)
y_train_clean = np.array(list(male_train_labels) + list(np.ones(len(female_train_clean))))
train_acc_clean, test_acc_clean, _ = train_classifier(X_train_clean, y_train_clean, X_test, y_test)
print("Classifier After Outlier Removal:")
print("Train Accuracy: {:.4f}".format(train_acc_clean))
print("Test Accuracy:  {:.4f}\n".format(test_acc_clean))
trimming_analysis(female_train_mod, male_train, male_train_labels, X_test, y_test)
print("\nSummary of Classifier Accuracies:")
print("Initial Data:            Train = {:.4f}, Test = {:.4f}".format(train_acc, test_acc))
print("After Outlier Injection: Train = {:.4f}, Test = {:.4f}".format(train_acc_mod, test_acc_mod))
print("After Outlier Removal:   Train = {:.4f}, Test = {:.4f}".format(train_acc_clean, test_acc_clean))
