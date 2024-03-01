import numpy as np

list1 = [[1, 2], [3, 4]]
list2 = [[5, 6], [7, 8]]

print(((np.array(list1) + np.array(list2)) / 2).tolist())
