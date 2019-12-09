import Data as d
import numpy as np
import matplotlib.pyplot as plt


plt.figure(1)
#plt.plot(d.yellow_V,d.yellow_I_norm,'.-y',label="Yellow")
#plt.plot(d.green_V,d.green_I_norm,'.-g',label="Green")
#plt.plot(d.VB_V,d.VB_I_norm,'.-',color="purple",label="VB")
#plt.plot(d.B_V,d.B_I_norm,'.-',color="blue",label="B")
#plt.plot(d.red_V,d.red_I_norm,'.-',color="red",label="Red")

plt.plot(d.yellow_V,d.yellow_I,'.-y',label="Yellow")
plt.plot(d.green_V,d.green_I,'.-g',label="Green")
plt.plot(d.VB_V,d.VB_I,'.-',color="purple",label="VB")
plt.plot(d.B_V,d.B_I,'.-',color="blue",label="B")
plt.plot(d.red_V,d.red_I,'.-',color="red",label="Red")

plt.xlabel("applied voltage (V)")
plt.ylabel("Normalised Current")
#plt.axis([-2.5, 2.5, -0.05, 0.4])
#plt.errorbar(d.yellow_V,d.yellow_I_norm, yerr=d.yellow_I_err,fmt='o', ecolor='yellow', capthick=2)
plt.legend()
plt.grid()
plt.show()
#%%
def linear_fit(x,y,i_0,i_1):
    return np.polyfit(x[i_0:i_1+1], y[i_0:i_1+1], 1)

x=[d.yellow_V,d.green_V,d.VB_V,d.B_V,d.red_V]
y=[d.yellow_I,d.green_I,d.VB_I,d.B_I,d.red_I]

i_0=27
i_1=36
i_0_1=40
i_1_1=len(d.yellow_V)

col_ind=4
yellow_line_1=linear_fit(x[col_ind],y[col_ind],i_0,i_1)
yellow_poly_1=np.poly1d(yellow_line_1)


yellow_line_2=linear_fit(x[col_ind],y[col_ind],i_0_1,i_1_1)
yellow_poly_2=np.poly1d(yellow_line_2)

def Linear_inter(m1,m2):
    return -(m2[1]-m1[1])/(m2[0]-m1[0])


VCO=[]
for i in range(len(x)):
    line_1=linear_fit(x[i],y[i],i_0,i_1)
    poly_1=np.poly1d(line_1)
    
    line_2=linear_fit(x[i],y[i],i_0_1,i_1_1)
    poly_2=np.poly1d(line_2)
    
    VCO.append(Linear_inter(poly_1,poly_2))

plt.figure()
plt.plot(x[col_ind],y[col_ind],'.-y')
plt.plot(np.linspace(-3.5,1,num=10),yellow_poly_1(np.linspace(-3.5,1,num=10)),color="green")
plt.plot(np.linspace(-1,9,num=10),yellow_poly_2(np.linspace(-1,9,num=10)),color="red")
plt.grid()
plt.show()

#%%

def grad_method(x,y):
    grad=1000
    old_grad=0
    index=0
    current_x=[]
    grad_array=[]
    while abs(grad-old_grad)>0.0001:
        old_grad=grad
        if index<len(x)-1:
            grad=(y[index+1]-y[index])/(x[index+1]-x[index])
            grad_array.append(grad)
            current_x.append(x[index+1])
            index=index+1
        else:
            break
    return current_x,grad_array

VCO1=[]
grad_array=[]
for i in range(len(x)):
    a,b=grad_method(x[i][31:49],y[i][31:49])
    VCO1.append(a)
    grad_array.append(b)

def interpolate_grad(x,y):
    m=(y[-1]-y[-2])/(x[-1]-x[-2])
    return (-y[-2]/m)+x[-2]

print(VCO)

VCO1_inter=[]
for i in range(len(VCO1)):
   VCO1_inter.append(interpolate_grad(VCO1[i],grad_array[i]))
print(VCO1_inter)

