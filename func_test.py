import timeit
from operator import itemgetter
import numpy as np

# 创建一个长度为5的列表
my_list = [10, 20, 30, 40, 50, 30, 409, 45, 656]
indices = [0, 2, 3, 1, 6]

# 方法1：列表推导式
def list_comprehension():
    return [my_list[i] for i in indices]

# 方法2：itemgetter
def use_itemgetter():
    return list(itemgetter(*indices)(my_list))

# 方法3：NumPy 数组
def use_numpy():
    np_array = np.array(my_list)
    return np_array[indices]

# 方法4：map 函数
def use_map():
    return list(map(my_list.__getitem__, indices))

# 方法5：简单循环
def use_loop():
    selected_elements = []
    for i in indices:
        selected_elements.append(my_list[i])
    return selected_elements

# 计时
iterations = 100000

print("List comprehension:", timeit.timeit(list_comprehension, number=iterations))
print("Itemgetter:", timeit.timeit(use_itemgetter, number=iterations))
print("NumPy:", timeit.timeit(use_numpy, number=iterations))
print("Map function:", timeit.timeit(use_map, number=iterations))
print("Simple loop:", timeit.timeit(use_loop, number=iterations))

