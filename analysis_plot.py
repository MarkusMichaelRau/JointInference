import numpy as np
import os
from matplotlib import pyplot as plt


directory = './result_new_constraints/'
list_dataset = []
list_sig8 = []
for filename in os.listdir(directory):
    print(filename)
    data = np.loadtxt(directory+filename)
    list_dataset.append(data)
    curr_list = filename.split("t1")
    curr_list = curr_list[1].split(".d")
    list_sig8.append(np.float(curr_list[0]))


list_sig8 = np.array(list_sig8)
list_dataset = np.array(list_dataset)

plt.plot(list_sig8, [np.min(el) for el in list_dataset])
plt.savefig('profile_likelihood.pdf')
