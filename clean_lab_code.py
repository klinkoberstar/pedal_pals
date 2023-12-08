import timm
import torch
import random
import torchvision
import numpy as np

from cleanlab import Datalab

from tqdm import tqdm
from matplotlib import pyplot as plt
from datasets import load_dataset
import pandas as pd

report = False 

# df = pd.read_csv('/Applications/Documents/Uchicago/2023_2024/1_Math_Foundations_ML/project/rs_million.csv')

# df['start_time_dt_fmt'] = pd.to_datetime(df['start_time'], format='%Y-%m-%dT%H:%M:%S.%f')
# df['hour'] = df['start_time_dt_fmt'].dt.hour
# df['min'] = df['start_time_dt_fmt'].dt.minute
# df['hour'] = df['start_time_dt_fmt'].dt
# df['time_in_hours'] = df['hour'] + df['min']/60
# df['trip_duration_min'] = df['trip_duration']/60
# df = df[df['trip_duration_min'] <= 60 * 10]

# df = df[]


num_classes = 10
means = [np.array([np.random.uniform(high=10), np.random.uniform(high=10)]) for i in range(num_classes)]
sigmas = [np.random.uniform(high=1) for i in range(num_classes)]
class_stats = list(zip(means, sigmas))

k = 10
num_samples = 1000

def generate_data_gradual_mean_shift():
    samples = []
    labels = []

    for i in range(num_samples):
        cls = np.random.choice(num_classes)
        mean, sigma = class_stats[cls]
        shift = 2 * i / num_samples # gradually increasing a mean shift for each gaussian
        sample = np.random.normal(mean + shift, sigma)
        samples.append(sample)
        labels.append(cls)
    samples = np.array(samples)
    labels = np.array(labels)
    
    plt.figure(figsize=(9,6.5), dpi=100)
    plt.scatter(samples[:,0], samples[:,1], c=np.arange(len(samples)))
    plt.colorbar(label='Datapoint index')

    plt.axis('off')
    plt.show()

    dataset = {'features': samples, 'labels': labels}
    return dataset


dataset = generate_data_gradual_mean_shift()
datalab = Datalab(dataset, label_name='labels')
datalab.find_issues(features=dataset['features'], issue_types={'non_iid': {}})
print('p-value =', datalab.get_info('non_iid')['p-value'])
if report:
    datalab.report()