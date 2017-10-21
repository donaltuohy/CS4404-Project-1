import numpy as np

x = np.matrix([
    [0,0,0], 
    [4,5,9],
    [3,10,6]
    ])

print("X: ", x, "\n")
min = np.amin(x, axis=0)
max = np.amax(x, axis=0)   

print("max: ", max, "\n")
print("min: ", min, "\n")

x = (x-min) / (max-min)

print("result: ", x)