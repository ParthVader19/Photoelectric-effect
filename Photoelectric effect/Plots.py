import Data as d
import numpy as np
import matplotlib.pyplot as plt
import Interpolation as inter

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
def linear_fit(x,y,i_0,i_1,err):
    return np.polyfit(x[i_0:i_1+1], y[i_0:i_1+1], 1,w=1/err[i_0:i_1+1])

x=[d.yellow_V,d.green_V,d.VB_V,d.B_V,d.red_V]
y=[d.yellow_I,d.green_I,d.VB_I,d.B_I,d.red_I]
y_err=[d.yellow_I_err,d.green_I_err,d.VB_I_err,d.B_I_err,d.red_I_err]

i_0=27
i_1=30
i_0_1=40
i_1_1=len(d.VB_V)-1

col_ind=4
yellow_line_1=linear_fit(x[col_ind],y[col_ind],i_0,i_1,err=y_err[col_ind])
yellow_poly_1=np.poly1d(yellow_line_1)


yellow_line_2=linear_fit(x[col_ind],y[col_ind],i_0_1,i_1_1,err=y_err[col_ind])
yellow_poly_2=np.poly1d(yellow_line_2)

plt.figure()
plt.plot(x[col_ind],y[col_ind],'.y')
plt.plot(np.linspace(-3.5,1,num=10),yellow_poly_1(np.linspace(-3.5,1,num=10)))
plt.plot(np.linspace(-1,9,num=10),yellow_poly_2(np.linspace(-1,9,num=10)),color="red")
plt.errorbar(x[col_ind],y[col_ind], yerr=y_err[col_ind],fmt='-y', ecolor='black', capthick=2)
plt.plot(x[col_ind][i_0],y[col_ind][i_0],'or')
plt.plot(x[col_ind][i_1],y[col_ind][i_1],'og')
plt.plot(x[col_ind][i_0_1],y[col_ind][i_0_1],'or')
plt.plot(x[col_ind][i_1_1-1],y[col_ind][i_1_1-1],'og')
plt.xlabel(r'$V_{CE}$ (V)')
plt.ylabel(r'Current (nA)')
plt.grid()
plt.show()
#%%
def Linear_inter(m1,m2):
    return -(m2[0]-m1[0])/(m2[1]-m1[1])

VCO=[]
i_data=[[30,35],[30,35],[28,34],[27,34],[27,30]]
i_data_end=[[42,len(d.yellow_V)-1],[42,len(d.green_V)-1],[44,len(d.VB_V)-1],[47,len(d.B_V)-1],[40,len(d.red_V)-1]]

poly2=[]
for i in range(len(x)):
    line_1=linear_fit(x[i],y[i],i_data[i][0],i_data[i][1],err=y_err[i])
    poly_1=np.poly1d(line_1)
    
    line_2=linear_fit(x[i],y[i],i_data_end[i][0],i_data_end[i][1],err=y_err[i])
    poly_2=np.poly1d(line_2)
    print(poly_2)
    poly2.append(poly_2)
    
    VCO.append(Linear_inter(poly_1,poly_2))



#%%

def grad_method(x,y):
    grad=1000
    old_grad=0
    index=0
    current_x=[]
    grad_array=[]
    while abs(grad-old_grad)>1:
#    while grad!=old_grad:
        old_grad=grad
        if index<len(x)-1:
            grad=(y[index+1]-y[index])/(x[index+1]-x[index])
            grad_array.append(grad)
            current_x.append(x[index+1])
            index=index+1
        else:
            break
#        print(grad-old_grad,x[index])
    return current_x,grad_array

VCO1=[]
grad_array=[]
for i in range(len(x)):
    a,b=grad_method(x[i][i_data[i][1]:i_data_end[i][0]],y[i][i_data[i][1]:i_data_end[i][0]])
#    print('-----')
#    print(i_data[i][1],i_data_end[i][0])
    if i==3:
        print(a,b)
        print('---------')
    VCO1.append(a)
    grad_array.append(b)
    print(a[-1])

def interpolate_grad(x,y):
    m=(y[-1]-y[-2])/(x[-1]-x[-2])
    return (-y[-2]/m)+x[-2]

print(VCO)
plt.plot(VCO1[0],grad_array[0])
#%%
roots=[]
for i in range(len(x)):
#    i=1
    f,x_domain,y_range=inter.cubicSpline(x[i],y[i])
    g=poly2[i](x_domain)
    index=0
    h=y_range-g
    root=100
    while h[index]>1e-2:
        index=index+1
        root=x_domain[index]
#        print(h[index])
    roots.append(root)
#%%
wavelenght=np.array([578.6,546.9,434.7,365.7])
c=3e8
plt.plot(c/wavelenght,VCO[:-1],'.')

