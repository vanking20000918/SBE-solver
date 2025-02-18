# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:15:07 2023

@author: Van king
"""

# Fast fourier transform 
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.fft import fft, fftfreq
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from constants import mathpi,au2eV,fs2au,h,e
from input_parameters import cycle,Order

#读取数据
time = np.loadtxt(os.path.join("data", "data.dat"),comments='#',usecols=0) # fs
data1 = np.loadtxt(os.path.join("data", "data.dat"),comments='#',usecols=1)
data2 = np.loadtxt(os.path.join("data", "data.dat"),comments='#',usecols=2)

# # Function to check if any array contains zero
# def contains_zero(array):
#     return 0 in array

# # Check each array for zero
# if contains_zero(data1) or contains_zero(data2):
#     print("One of the data contains zero. Exiting.")
#     sys.exit()
    
Photonenergy = mathpi * 2 / cycle * au2eV #光子能量，单位eV 

#定义采样周期、采样数、总阶数
sampling_period = time[1]-time[0] # 采样周期，单位fs
sampling_number = time.size #采样数
Total_order = 0.5 / sampling_period * (h/1e-15/e) / (Photonenergy) # 计算采样周期对应的总阶数
 
#选择窗函数并处理
window = np.hamming(data1.size) # 可以选择窗口函数如np.bartlett, np.blackman, np.hamming, np.hanning, 
data1_window = data1 * window
data2_window = data2 * window

#计算傅里叶变换
complex_array1 = fft(data1_window, n=data1.size, norm=None)
complex_array2 = fft(data2_window, n=data2.size, norm=None)

# 得到分解波的频率序列
freqs = fftfreq(sampling_number, sampling_period) 

#对频率进行单位换算
freqs = freqs * (h/1e-15/e) / (Photonenergy) # 将单位从1/s换算成光子能量

# 计算原始信号的相位、幅度、振幅
Power1 = np.abs(complex_array1)**2 # 以均方振幅（MSA）表示功率密度 
Power2 = np.abs(complex_array2)**2 # 以均方振幅（MSA）表示功率密度 

# 绘制原始信号与傅里叶分析图
fig, axs = plt.subplots(nrows=2, figsize=(7,7), constrained_layout=True, num=8)
plt.rcParams['font.family'] = 'Times New Roman'

#plt.style.use(['science', 'no-latex' ]) # 采用SciencePlots格式

# 绘制原始信号
axs[0].plot(time / cycle * fs2au, data1**2, c='red', label="Intraband")
axs[0].plot(time / cycle * fs2au, data2**2, c='blue', label="Interband")
axs[0].set_title('Signal')
axs[0].set_xlabel('Time(cycles)')
axs[0].set_ylabel('Intensity (arb.unit)')
axs[0].xaxis.set_major_formatter(ScalarFormatter()) #设置刻度格式
axs[0].xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
axs[0].yaxis.set_major_formatter(ScalarFormatter())
axs[0].yaxis.set_minor_formatter(FormatStrFormatter('%1.0f'))
axs[0].tick_params(direction='in', which='both')
axs[0].legend(loc='upper right', fontsize='medium')
#绘制傅里叶分析图
axs[1].plot(freqs[freqs > 0], np.log10(Power1[freqs > 0]), c='red', label="Intraband")
axs[1].plot(freqs[freqs > 0], np.log10(Power2[freqs > 0]), c='blue', label="Interband")
axs[1].set_title('Fourier Transform Analysis')
axs[1].set_xlabel('Frequency(Photon energy)')
axs[1].set_ylabel('Log(power)\n(arb.unit)')#设置刻度和标签格式
axs[1].set_xlim(0, Order)
axs[1].set_xticks(np.arange(1, Order+1, 2))
x_min, x_max = axs[1].get_xlim() # 根据x轴刻度范围调整y轴刻度范围
#y_min, y_max = np.log10(Power[(freqs > 0) & (freqs <= x_max)]).min(), np.log10(Power[(freqs > 0) & (freqs <= x_max)]).max()
y_min_1, y_max_1 = np.log10(Power1[(freqs > 0) & (freqs <= x_max)]).min(), np.log10(Power1[(freqs > 0) & (freqs <= x_max)]).max()
y_min_2, y_max_2 = np.log10(Power2[(freqs > 0) & (freqs <= x_max)]).min(), np.log10(Power2[(freqs > 0) & (freqs <= x_max)]).max()
y_ticks = np.arange(np.floor(min(y_min_1,y_min_2)), np.ceil(max(y_max_1,y_max_2)) + 2)
axs[1].set_ylim(y_ticks[0], y_ticks[-1])
axs[1].set_yticks(y_ticks)
axs[1].xaxis.set_major_formatter(ScalarFormatter()) 
axs[1].xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
axs[1].yaxis.set_major_formatter(ScalarFormatter())
axs[1].yaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
axs[1].tick_params(direction='in', which='both')
axs[1].legend(loc='upper right', fontsize='medium')
#插入文本
plt.figtext(0.75, 0.5, f"Photon energy={Photonenergy:.3f}eV")

#在傅里叶变换分析图中设置参考线
x_ref = np.arange(1, Order+1, 1)
for xi in x_ref:
    axs[1].axvline(x=xi, color='black', linestyle='--')
#plt.show()
fig.savefig('Fast fourier analysis.png')