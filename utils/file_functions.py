# Saving and loading of files
import numpy as np
import os.path

# compression of 1.1MB for array of 100 samples ...

def save_compressed(array, name_without_npz):
    np.savez_compressed(name_without_npz + ".npz", a=array)

def load_compressed(name_without_npz):
    array = np.load(name_without_npz + ".npz")['a']
    return array

def file_exists(path):
    return os.path.exists(path)


"""
tmp = np.load("../data/saved_impulses_100.npy")

save_compressed(tmp, "../data/saved_impulses_100")
tmp2 = load_compressed("../data/saved_impulses_100")

equal = np.array_equal(tmp, tmp2)
print("Equal?", equal)
"""

#tmp0 = load_compressed("../data/saved_impulses_15")
#print(tmp0.shape)
#print(np.random.rand(4, 40, 1025).shape)