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
#    print(poly_2)
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
#    print(a[-1])

def interpolate_grad(x,y):
    m=(y[-1]-y[-2])/(x[-1]-x[-2])
    return (-y[-2]/m)+x[-2]

print(VCO)
#plt.plot(VCO1[0],grad_array[0])
#%%
def cub_spl(x,y):
    roots=[]
    x_spline=[]
    y_spline=[]
    for i in range(len(x)):
        f,x_domain,y_range=inter.cubicSpline(x[i][i_data[i][1]:i_data_end[i][0]+6],y[i][i_data[i][1]:i_data_end[i][0]+6])
        x_spline.append(x_domain)
        y_spline.append(y_range)
        g=poly2[i](x_domain)
        index=0
        h=y_range-g
        root=100
        while h[index]>1e-2:
            if index<len(x_domain)-1:
                index=index+1
                root=x_domain[index]
            else:
                break
        roots.append(root)
    return roots,x_spline,y_spline

roots,x_spline,y_spline=cub_spl(x,y)

#%%
wavelenght=np.array([578.6,546.9,434.7,365.7])*1e-9
c=3e8

line_inter=np.polyfit(c/wavelenght,VCO[:-1],1)
line_inter_poly=np.poly1d(line_inter)

line_grad=np.polyfit(c/wavelenght,roots[:-1],1)
line_grad_poly=np.poly1d(line_grad)

plt.figure()
plt.plot(c/wavelenght,VCO[:-1],'.r')
plt.plot(c/wavelenght,roots[:-1],'.g')
plt.plot(np.linspace(3e14,1e15,num=10),line_inter_poly(np.linspace(3e14,1e15,num=10)),color="red",label='Interpolation Method')
plt.plot(np.linspace(3e14,1e15,num=10),line_grad_poly(np.linspace(3e14,1e15,num=10)),color="green",label='Trending Method')
plt.legend()
plt.grid()
plt.show()

def analysis(x):
    return x[1]*(1.6e-19),-x[0]

inter_h,inter_work=analysis(line_inter_poly)
grad_h,grad_work=analysis(line_grad_poly)

#def spline_err(x_real,y_real,x_in,y_in):
#    f=0
#    err_y=[]
#    for j in range(len(y_real)):
#        prod=1
#        for k in range(len(x_real)):
#            if k!=j:
#              prod=prod*(x_real[j]-x_real[k])
#        f=f+y_real[j]/prod
##    print(f)
#    for i in range(len(x_in)):
#        prod_2=1
#        for b in range(len(x_real)):
#            prod_2=prod_2*(x_in[i]-x_real[b])
##        print(prod_2)
#        err_y.append(abs(prod_2*f))
#    return err_y
#
#err_spline=spline_err(x[0][i_data[0][1]:i_data_end[0][0]],y[0][i_data[0][1]:i_data_end[0][0]],x_spline[0],y_spline[0])

plt.figure()
plt.plot(x_spline[0],y_spline[0],'.',color="red")
#plt.errorbar(x_spline[0],y_spline[0], yerr=err_spline,fmt='-y', ecolor='black', capthick=2)
plt.plot(x[0],y[0],'.')
plt.show()
print(inter_h,inter_work)
print(grad_h,grad_work)
#plt.figure()
##plt.plot(x_spline[0],err_spline)


