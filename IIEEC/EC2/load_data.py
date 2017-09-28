import numpy as np
data = np.loadtxt("HiSeqV2_PANCAN.txt", skiprows=1, usecols=range(1, 1219))
data = data.T # make the data matrix be of shape (samples X features) 
