import Data as d
import numpy as np
import matplotlib.pyplot as plt


plt.figure(1)
plt.plot(d.yellow_V,d.yellow_I_norm,'.-y',label="Yellow")
plt.plot(d.green_V,d.green_I_norm,'.-g',label="Green")
plt.plot(d.VB_V,d.VB_I_norm,'.-',color="purple",label="VB")
plt.plot(d.B_V,d.B_I_norm,'.-',color="blue",label="B")
plt.plot(d.red_V,d.red_I_norm,'.-',color="red",label="Red")
plt.xlabel("applied voltage (V)")
plt.ylabel("Normalised Current")
#plt.axis([-2.5, 2.5, -0.05, 0.4])
#plt.errorbar(d.yellow_V,d.yellow_I_norm, yerr=d.yellow_I_err,fmt='o', ecolor='yellow', capthick=2)
plt.legend()
plt.grid()
plt.show()



