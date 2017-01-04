import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

x = np.arange(0, 20, 0.25)
F = 10*(1 + 3.33 * 0.50*np.sqrt(x) - .35*x)
fig = plt.figure()
ax = fig.add_subplot(111)
# ax = fig.add_subplot(111, projection='3d')
ax.plot(x, F)
plt.xlim([0,20])
plt.ylim([0,80])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Meshgrid")
plt.show()
