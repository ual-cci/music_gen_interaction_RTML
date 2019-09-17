# Saving and loading of files
import numpy as np

# compression of 0.2 for 15 files -

def save_compressed(array, name_without_npz):
    np.savez_compressed(name_without_npz + ".npz", a=array)

def load_compressed(name_without_npz):
    array = np.load(name_without_npz + ".npz")['a']
    return array

"""
tmp = np.load("../data/saved_impulses_15.npy")

save_compressed(tmp, "../data/saved_impulses_15")
tmp2 = load_compressed("../data/saved_impulses_15")

equal = np.array_equal(tmp, tmp2)
print("Equal?", equal)
"""