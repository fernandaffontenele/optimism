from matplotlib import pyplot as plt
import numpy as np

plotData = np.load('arch_bc_Fd.npz')

reaction = plotData['force']
displacement = plotData['displacement']

plt.plot(displacement, reaction)
plt.show()
