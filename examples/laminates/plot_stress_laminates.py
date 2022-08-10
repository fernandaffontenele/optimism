import numpy as np
import numpy as onp

stress_data  = onp.loadtxt("stress_data.txt")
data_size = len(stress_data[:,0])
S_11 = np.zeros((int(data_size/3),1))
ind = 0
for i in range(0,data_size,3):
    S_11[ind] = stress_data[i,0]
    ind = ind + 1


