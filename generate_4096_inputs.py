import numpy as np

m = 4096
n = 4096
k = 4096
# l = np.random.randint(low=0, high=255, size=(1280, 4096), dtype=np.uint8)
l = np.random.randint(low=0, high=255, size=(m, k), dtype=np.uint8)
r = np.random.randint(low=0, high=255, size=(k, n), dtype=np.uint8)


l.tofile("lhs_4096.bin")
r.tofile("rhs_4096.bin")

# 2048 4096 14336
# 2048 14336 4096

# 2048 128256 4096