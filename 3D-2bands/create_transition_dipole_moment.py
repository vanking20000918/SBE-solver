# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:41:18 2023

@author: Van king
"""

import numpy as np
from input_parameters import num_kpoints

dim = 3
data = np.zeros((num_kpoints[-1], dim * num_kpoints[0] * num_kpoints[1]*2))

""""
我们对'transition_dipole_moment.dat'文件的格式设置如下：
对于任意n维度（n=1,2,3）下的k1（*k2*k3）网格点，这个文件共有k3个行（最后一个k网格），总共有
2*n*k1(*k2)，即是网格点*维度的两倍。这是因为我们把跃迁矩阵元的实、虚部分开储存。奇数列储存实部，
偶数列储存虚部。行的顺序按照k点从（-pi/a,pi/a），行的顺序按照维度的顺序排列。
比如，我有一个400*300*200的三维网格点，那储存文件的格式为：200行*（2*3*400*300=720000）列，
前2*400*300列对应第一个维度（x方向）的跃迁矩阵元，其中奇数列是对应的实部，偶数列对应的是虚部，
前400列对应第二个k点网格维度第一个，第二个400列对应第二个k点网格维度第二个，即400+400+400...(300个)。
类比第二个2*400*300列对应第二个维度（y方向）的跃迁矩阵元，第三个2*400*300列对应第三个维度（z方向）
的跃迁矩阵元。

"""

# you can assign values to you data values like the example
# 设置前一百列的奇数列为 3.46
data[:, 0:100:2] = 3.46

# 第二个一百列的奇数列同样设置为 3.46
data[:, 100:200:2] = 3.46

# 第三个一百列的奇数列设置为 3.94
data[:, 200:300:2] = 3.94

text_file_path = 'transition_dipole_moment.dat'
np.savetxt(text_file_path, data)





