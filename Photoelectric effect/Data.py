import numpy as np

yellow_V,yellow_I,yellow_I_err=np.loadtxt("Yellow_data_2.csv", delimiter=',', unpack=True)
green_V,green_I,green_I_err=np.loadtxt("Green_data_2.csv", delimiter=',', unpack=True)
VB_V,VB_I,VB_I_err=np.loadtxt("VB_data_2.csv", delimiter=',', unpack=True)
B_V,B_I,B_I_err=np.loadtxt("B_data_2.csv", delimiter=',', unpack=True)
red_V,red_I,red_I_err=np.loadtxt("Red_data_2.csv", delimiter=',', unpack=True)

def normal(x):
    return (x-min(x))/(max(x)-min(x))

yellow_I_norm=normal(yellow_I)
green_I_norm=normal(green_I)
VB_I_norm=normal(VB_I)
B_I_norm=normal(B_I)
red_I_norm=normal(red_I)
