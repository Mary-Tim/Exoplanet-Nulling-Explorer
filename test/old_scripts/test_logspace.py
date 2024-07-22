import numpy as np
from matplotlib import pyplot as plt

def log(low, high, num):
    #exponential_array = (np.exp(np.linspace(0, np.log(2), num))-1.)
    #exponential_array = exponential_array * (high-low) + low
    log_array = np.log(np.linspace(1., np.e, num)) * (high-low) + low
    return log_array

#print(log(5, 25, 20, 2))
low = 30
high = 50
bins = 50
plt.plot(np.linspace(low, high, bins), log(low, high, bins), c='b')
plt.show()