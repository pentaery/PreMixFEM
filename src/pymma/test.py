import numpy as np

xval = np.fromfile('/home/pentaery/projects/PreMixFEM/src/pymma/data/xval.bin', dtype=np.float64,order='>')

print(xval)
print(xval.shape)