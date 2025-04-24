import numpy
import statistics
import matplotlib.pyplot as plt

samples_cnt = 1000
req_female_mean = 152
req_male_mean = 166
req_sd = 5

male_samples = numpy.random.normal(loc = req_male_mean, scale = req_sd, size = samples_cnt);
female_samples = numpy.random.normal(loc = req_female_mean, scale = req_sd, size = samples_cnt);

min_error = 100
best_threshold = 0

# if height is greater than threshold then person is male else person is female
for threshold in range(1, 181):
    mismatch_cnt = 0
    
    for height in male_samples :
        if height < threshold :
            mismatch_cnt = mismatch_cnt + 1
    
    for height in female_samples :
        if height > threshold :
            mismatch_cnt = mismatch_cnt + 1
    
    # percentage of error
    error_per = (mismatch_cnt * 100) / (samples_cnt * 2);
    # min_error = min(error_per, min_error)
    if min_error > error_per : 
        min_error = error_per
        best_threshold = threshold

print(min_error);
print(best_threshold);

# using gaussian probability distribution
actual_male_mean = statistics.mean(male_samples)
actual_female_mean = statistics.mean(female_samples)

actual_male_sd = statistics.stdev(male_samples);
actual_female_sd = statistics.stdev(female_samples);

male_nd = statistics.NormalDist(mu=actual_male_mean, sigma=actual_male_sd)
female_nd = statistics.NormalDist(mu=actual_female_mean, sigma=actual_female_sd)

mismatch_cnt = 0;
for height in male_samples : 
    prob_male = male_nd.pdf(height)
    prob_female = female_nd.pdf(height)

    if prob_female > prob_male : 
        mismatch_cnt = mismatch_cnt + 1

for height in female_samples : 
    prob_male = male_nd.pdf(height)
    prob_female = female_nd.pdf(height)

    if prob_male > prob_female : 
        mismatch_cnt = mismatch_cnt + 1

error = (mismatch_cnt * 100) / (samples_cnt * 2)
print(error)

# plot data on graph
plt.hist(male_samples, 100)
# plt.show()

plt.hist(female_samples, 100)
plt.show()