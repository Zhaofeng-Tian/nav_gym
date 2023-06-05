import numpy as np

ctr = 0
buffer_size = 10
n = 8
index = ctr % buffer_size
obs_shape = (2,)
print(*obs_shape)
buffer = np.zeros((buffer_size, *obs_shape))
print(buffer.shape)
obs = np.ones((n, *obs_shape))
a = buffer.copy()
a[index:index+n] = obs
print(a)
obs2 = 2*obs
ctr+=n
index = ctr % buffer_size
if index + n > buffer_size:
    a[index:] = obs2[:buffer_size-index]
    ctr += buffer_size-index
else:
    a[index:index+n]= obs2
print(a)
print(ctr)