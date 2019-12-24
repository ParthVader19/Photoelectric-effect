import Data as d
import numpy as np
import matplotlib.pyplot as plt
import Interpolation as inter

#ND filters

ND_203_V,ND_203_I,ND_203_I_err=np.loadtxt("(green 203B).csv", delimiter=',', unpack=True)
ND_204_V,ND_204_I,ND_204_I_err=np.loadtxt("(green 204B).csv", delimiter=',', unpack=True)
ND_205_V,ND_205_I,ND_205_I_err=np.loadtxt("(green 205B).csv", delimiter=',', unpack=True)
ND_210_V,ND_210_I,ND_210_I_err=np.loadtxt("(green 210B).csv", delimiter=',', unpack=True)

pow_val,pow_err=np.loadtxt("QE.csv", delimiter=',', unpack=True)

wavelenght=np.array([578.6,546.9,434.7,406.6,365.7,691.3])
wavelenght_err=np.array([2.04969,0.672794,0.344067,3.60684,3.97668,9.04574])*1e-9
c=3e8

colour=['y','g',"purple","grey","blue","red"]
nd_filters=['203B','204B','205B','210B']

x=[d.yellow_V,d.green_V,d.VB_V,d.VA_V,d.B_V,d.red_V]
y=[d.yellow_I,d.green_I,d.VB_I,d.VA_I,d.B_I,d.red_I]
y_err=[d.yellow_I_err,d.green_I_err,d.VB_I_err,d.VA_I_err,d.B_I_err,d.red_I_err]

x_nd=[ND_203_V,ND_204_V,ND_205_V,ND_210_V]
y_nd=[ND_203_I,ND_204_I,ND_205_I,ND_210_I]
y_err_nd=[ND_203_I_err,ND_204_I_err,ND_205_I_err,ND_210_I_err]

#plt.figure()
#plt.plot(d.yellow_V,d.yellow_I_norm,'.-y',label="Yellow")
#plt.plot(d.green_V,d.green_I_norm,'.-g',label="Green")
#plt.plot(d.VB_V,d.VB_I_norm,'.-',color="purple",label="VB")
#plt.plot(d.B_V,d.B_I_norm,'.-',color="blue",label="B")
#plt.plot(d.red_V,d.red_I_norm,'.-',color="red",label="Red")

#plt.plot(d.yellow_V,d.yellow_I,'.-y',label="Yellow")
#plt.plot(d.green_V,d.green_I,'.-g',label="Green")
#plt.plot(d.VB_V,d.VB_I,'.-',color="purple",label="VB")
#plt.plot(d.B_V,d.B_I,'.-',color="blue",label="B")
#plt.plot(d.red_V,d.red_I,'.-',color="red",label="Red")
plt.figure()
for i in range(len(x)):
    plt.plot(x[i],y[i],'.-',color=colour[i],label=str(wavelenght[i])+'nm')
    plt.errorbar(x[i],y[i], yerr=y_err[i],fmt='.-',color=colour[i], ecolor='black', capthick=2)
plt.xlabel("Applied Voltage (V)")
plt.ylabel("Current (nA)")
#plt.ylabel("Normalised Current")
plt.legend()
plt.grid()
plt.show()

wavelenght=np.array([578.6,546.9,434.7,406.6,365.7,691.3])*1e-9
freq_err=(wavelenght_err/wavelenght)*(c/wavelenght)
freq=c/wavelenght

plt.figure()
for i in range(len(x_nd)):
    plt.plot(x_nd[i],y_nd[i],'.-',color=colour[i],label=nd_filters[i])
    plt.errorbar(x_nd[i],y_nd[i], yerr=y_err_nd[i],fmt='.-',color=colour[i], ecolor='black', capthick=2)
plt.xlabel("Applied Voltage (V)")
plt.ylabel("Current (nA)")
#plt.ylabel("Normalised Current")
plt.legend()
plt.grid()
plt.show()
#%%
def linear_fit(x,y,i_0,i_1,err):
    return np.polyfit(x[i_0:i_1+1], y[i_0:i_1+1], 1,w=1/err[i_0:i_1+1],cov=True)

col_ind=3
i_0=13
i_1=17
i_0_1=25
i_1_1=len(x[col_ind])-1


yellow_line_1=linear_fit(x[col_ind],y[col_ind],i_0,i_1,err=y_err[col_ind])
yellow_poly_1=np.poly1d(yellow_line_1[0])


yellow_line_2=linear_fit(x[col_ind],y[col_ind],i_0_1,i_1_1,err=y_err[col_ind])
yellow_poly_2=np.poly1d(yellow_line_2[0])

plt.figure()
#plt.plot(x[col_ind],y[col_ind],'.y')
plt.plot(np.linspace(-3.5,1,num=10),yellow_poly_1(np.linspace(-3.5,1,num=10)),linewidth=2,label="Extrapolating Linear Section")
plt.plot(np.linspace(-1,9,num=10),yellow_poly_2(np.linspace(-1,9,num=10)),color="red",linewidth=2,label="Extrapolating Offset Section")
plt.errorbar(x[col_ind],y[col_ind], yerr=y_err[col_ind],fmt='--.',color="black", ecolor='black', capthick=2,label="Yellow Light Data")
plt.plot(x[col_ind][i_0],y[col_ind][i_0],'or')
plt.plot(x[col_ind][i_1],y[col_ind][i_1],'og')
plt.plot(x[col_ind][i_0_1],y[col_ind][i_0_1],'or')
plt.plot(x[col_ind][i_1_1-1],y[col_ind][i_1_1-1],'og')
#plt.axis([-5, 5, -5, 175])
plt.xlabel(r'Applied Voltage (V)')
plt.ylabel(r'Current (nA)')
plt.legend()
plt.grid()
plt.show()

    
#%%
def Linear_inter(m1,m2):
    return -(m2[0]-m1[0])/(m2[1]-m1[1])


def linear_err(line1,line2):
    c1=line1[0][1]
    c2=line2[0][1]
    m1=line1[0][0]
    m2=line2[0][0]
    
    M=m2-m1
    M_err=np.sqrt((line1[1][0][0])**2 + (line2[1][0][0])**2)
    C2=-c2/M
    C1=c1/M
    C2_err=C2*np.sqrt((line2[1][1][1]/c2)**2 +(M_err/M)**2)
    C1_err=C1*np.sqrt((line1[1][1][1]/c1)**2 +(M_err/M)**2)
    
    return C2+C1,np.sqrt(C1_err**2 + C2_err**2)

VCO=[]
VCO_err=[]
i_data=[[30,35],[30,35],[28,34],[13,17],[27,34],[27,30]]
i_data_end=[[42,len(d.yellow_V)-1],[42,len(d.green_V)-1],[44,len(d.VB_V)-1],[25,len(x[col_ind])-1],[47,len(d.B_V)-1],[40,len(d.red_V)-1]]

poly2=[]
for i in range(len(x)-1):
    line_1=linear_fit(x[i],y[i],i_data[i][0],i_data[i][1],err=y_err[i])
    poly_1=np.poly1d(line_1[0])
    
    line_2=linear_fit(x[i],y[i],i_data_end[i][0],i_data_end[i][1],err=y_err[i])
    poly_2=np.poly1d(line_2[0])
#    print(poly_2)
    poly2.append(poly_2)
    V,V_err=linear_err(line_1,line_2)
    #VCO.append(Linear_inter(poly_1,poly_2))
    VCO.append(V)
    VCO_err.append(V_err)



##%%
#
#def grad_method(x,y):
#    grad=1000
#    old_grad=0
#    index=0
#    current_x=[]
#    grad_array=[]
#    while abs(grad-old_grad)>1:
##    while grad!=old_grad:
#        old_grad=grad
#        if index<len(x)-1:
#            grad=(y[index+1]-y[index])/(x[index+1]-x[index])
#            grad_array.append(grad)
#            current_x.append(x[index+1])
#            index=index+1
#        else:
#            break
##        print(grad-old_grad,x[index])
#    return current_x,grad_array
#
#VCO1=[]
#grad_array=[]
#for i in range(len(x)):
#    a,b=grad_method(x[i][i_data[i][1]:i_data_end[i][0]],y[i][i_data[i][1]:i_data_end[i][0]])
##    print('-----')
##    print(i_data[i][1],i_data_end[i][0])
#    if i==3:
#        print(a,b)
#        print('---------')
#    VCO1.append(a)
#    grad_array.append(b)
##    print(a[-1])
#
#def interpolate_grad(x,y):
#    m=(y[-1]-y[-2])/(x[-1]-x[-2])
#    return (-y[-2]/m)+x[-2]
#
#print(VCO)
##plt.plot(VCO1[0],grad_array[0])
#%%
def cub_spl(x1,y1):
    roots=[]
    x_spline=[]
    y_spline=[]
    x_spline_range=[]
    for i in range(len(x1)):
        f,x_domain,y_range=inter.cubicSpline(x1[i][i_data[i][1]:i_data_end[i][1]],y1[i][i_data[i][1]:i_data_end[i][1]])
        x_spline.append(x_domain)
        y_spline.append(y_range)
        g=poly2[i](x_domain)
        index=0
        h=y_range-g
        root=100
        while h[index]>0.5e-1:
            if index<len(x_domain)-1:
                index=index+1
                root=x_domain[index]
            else:
                break
        for j in range(len(x1[i])-1):
            if root>x1[i][j] and root<x1[i][j+1]:
                #x_spline_range.append([x[i][j],x[i][j+1],x[i][j+1]-x[i][j]])
                x_spline_range.append(x1[i][j+1]-x1[i][j])
                
        roots.append(root)
    return roots,x_spline,y_spline,x_spline_range

poly_nd_1=[]
for i in range(len(x_nd)):
#    line_1=linear_fit(x_nd[i],y_nd[i],i_data[i][0],i_data[i][1],err=y_err[i])
#    poly_1=np.poly1d(line_1)
#    
    line_nd=linear_fit(x_nd[i],y_nd[i],16,len(y_nd[i]),err=y_err_nd[i])
    poly_nd=np.poly1d(line_nd[0])
#    print(poly_nd)
    poly_nd_1.append(poly_nd)
    
#    VCO.append(Linear_inter(poly_1,poly_2))
 #%%   
def cub_spl_nd(x1,y1):
    roots=[]
    x_spline=[]
    y_spline=[]
    x_spline_range=[]
    for i in range(len(x1)):
        f,x_domain,y_range=inter.cubicSpline(x1[i],y1[i])
        x_spline.append(x_domain)
        y_spline.append(y_range)
        g=poly_nd_1[i](x_domain)
        index=0
        h=y_range-g
        root=100
        while h[index]>1e-1:
            if index<len(x_domain)-1:
                index=index+1
                root=x_domain[index]
            else:
                break
        for j in range(len(x1[i])-1):
            if root>x1[i][j] and root<x1[i][j+1]:
                #x_spline_range.append([x[i][j],x[i][j+1],x[i][j+1]-x[i][j]])
                x_spline_range.append(x1[i][j+1]-x1[i][j])
                
        roots.append(root)
    return roots,x_spline,y_spline,x_spline_range
#%%
roots,x_spline,y_spline,x_spline_range=cub_spl(x[:-1],y[:-1])
roots_nd,x_spline_nd,y_spline_nd,x_spline_range_nd=cub_spl_nd(x_nd,y_nd)



line_inter=np.polyfit(freq[:-1],VCO,1,w=1/np.array(VCO_err),cov=True)
line_inter_poly=np.poly1d(line_inter[0])

line_grad=np.polyfit(freq[:-1],roots,1,w=1/np.array(x_spline_range),cov=True)
line_grad_poly=np.poly1d(line_grad[0])

plt.figure()
plt.plot(freq[:-1],VCO,'.r')
plt.errorbar(freq[:-1],VCO, xerr=freq_err[:-1],yerr=VCO_err,fmt='.r',ecolor='black', capthick=2)

plt.plot(freq[:-1],roots,'.g')
plt.errorbar(freq[:-1],roots, xerr=freq_err[:-1],yerr=x_spline_range,fmt='.g',ecolor='black', capthick=2)

plt.plot(np.linspace(3e14,1e15,num=10),line_inter_poly(np.linspace(3e14,1e15,num=10)),color="red",label='Interpolation Method')
plt.plot(np.linspace(3e14,1e15,num=10),line_grad_poly(np.linspace(3e14,1e15,num=10)),color="green",label='Trending Method')
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'$V_{CE}$ (V)')
plt.legend()
plt.grid()
plt.show()
#%%
def analysis(x):
    return x[0][0]*(1.6e-19),-x[0][1],np.sqrt(x[1][0][0])*(1.6e-19),np.sqrt(x[1][1][1])

inter_h,inter_work,inter_h_err,inter_work_err=analysis(line_inter)
grad_h,grad_work,grad_h_err,grad_work_err=analysis(line_grad)

print('Linear')
print(inter_h,inter_h_err)
print(inter_work,inter_work_err)
print('Cubic')
print(grad_h,grad_h_err)
print(grad_work,grad_work_err)

#%%
QE=[]
QE_err=[]
h=6.63e-34
e=1.6e-19
for i in range(len(x)):
    for j in range(len(x[i])):
        if x[i][j]==0:
            A=(y[i][j]*1e-9/e)/(pow_val[i]/(h*freq[i]))
            QE.append(A)
            err1=np.sqrt((y_err[i][j]/y[i][j])**2 +(freq_err[i]/freq[i])**2)
            QE_err.append(A*np.sqrt((err1/(y[i][j]*freq[i]))**2 + (pow_err[i]/pow_val[i])**2))
            

plt.figure()
#plt.plot(freq,QE,'.')
plt.errorbar(freq,QE, xerr=freq_err,yerr=QE_err,fmt='.g',ecolor='black', capthick=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('External Quantum Efficiency')
plt.grid()
plt.show()


