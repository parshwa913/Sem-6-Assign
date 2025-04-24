import numpy as np
from scipy.stats import norm, multivariate_normal

np.random.seed(42)
n = 1000

# Uncorrelated Features (Height and Haemoglobin)
height_f = np.random.normal(152, 5, n)
hb_f = np.random.normal(13, 1, n)
height_m = np.random.normal(166, 5, n)
hb_m = np.random.normal(14, 1, n)

mean_height_f = np.mean(height_f)
std_height_f = np.std(height_f)
mean_hb_f = np.mean(hb_f)
std_hb_f = np.std(hb_f)

mean_height_m = np.mean(height_m)
std_height_m = np.std(height_m)
mean_hb_m = np.mean(hb_m)
std_hb_m = np.std(hb_m)

def likelihood_uncorrelated(h, hb, gender):
    if gender == 'F':
        lh = norm.pdf(h, loc=mean_height_f, scale=std_height_f)
        lhb = norm.pdf(hb, loc=mean_hb_f, scale=std_hb_f)
    else:
        lh = norm.pdf(h, loc=mean_height_m, scale=std_height_m)
        lhb = norm.pdf(hb, loc=mean_hb_m, scale=std_hb_m)
    return lh * lhb

correct_uncorr = 0
for i in range(n):
    if likelihood_uncorrelated(height_f[i], hb_f[i], 'F') > likelihood_uncorrelated(height_f[i], hb_f[i], 'M'):
        correct_uncorr += 1
for i in range(n):
    if likelihood_uncorrelated(height_m[i], hb_m[i], 'M') > likelihood_uncorrelated(height_m[i], hb_m[i], 'F'):
        correct_uncorr += 1
accuracy_uncorr = correct_uncorr / (2*n)
print("Uncorrelated Features Accuracy: {:.2f}%".format(accuracy_uncorr*100))

# Correlated Features (Height and Weight)
def generate_correlated(mean_h, std_h, mean_w, std_w, corr, n):
    cov_hw = corr * std_h * std_w
    cov_matrix = [[std_h**2, cov_hw], [cov_hw, std_w**2]]
    means = [mean_h, mean_w]
    return np.random.multivariate_normal(means, cov_matrix, n)

female_corr = generate_correlated(152, 5, 60, 7, 0.6, n)
male_corr = generate_correlated(166, 5, 70, 7, 0.6, n)

mean_vec_f = np.mean(female_corr, axis=0)
cov_f = np.cov(female_corr, rowvar=False)
mean_vec_m = np.mean(male_corr, axis=0)
cov_m = np.cov(male_corr, rowvar=False)

def likelihood_correlated(x, gender):
    if gender == 'F':
        return multivariate_normal.pdf(x, mean=mean_vec_f, cov=cov_f)
    else:
        return multivariate_normal.pdf(x, mean=mean_vec_m, cov=cov_m)

correct_corr = 0
for i in range(n):
    x = female_corr[i]
    if likelihood_correlated(x, 'F') > likelihood_correlated(x, 'M'):
        correct_corr += 1
for i in range(n):
    x = male_corr[i]
    if likelihood_correlated(x, 'M') > likelihood_correlated(x, 'F'):
        correct_corr += 1
accuracy_corr = correct_corr / (2*n)
print("Correlated Features Accuracy: {:.2f}%".format(accuracy_corr*100))
