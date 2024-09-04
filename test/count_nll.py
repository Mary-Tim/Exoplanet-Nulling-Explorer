import sys
sys.path.append('..')

import numpy as np
from matplotlib import pyplot as plt
from nullingexplorer.io import FitResult

result = FitResult.load("results/Job_20240903_140611/result_earth.hdf5")
scan_nll = result.get_item('scan_nll')

fig, ax = plt.subplots()
line = np.arange(0, len(scan_nll))
scat = ax.scatter(line, scan_nll, s=30)
ax.set_xlabel("Task")
ax.set_ylabel("NLL")

print(np.count_nonzero(scan_nll<15370))

plt.show()
