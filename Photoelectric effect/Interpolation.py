import numpy as np
import matplotlib.pyplot as plt
import LUdecomp as LU


def simpleInter(x,y):
    inter_x=[]#the interpolated x positions
    inter_y=[]#the interpolated y positions
    for i in range(len(x)-1):
        a= np.linspace(x[i],x[i+1],5,endpoint=True)#5 points between x[i] and x[i+1]
        for j in range(len(a)):#Simple Interpolation: a straight line is constructed between x[i] and x[i+1]
            y_a=((x[i+1]-a[j])*y[i]+(a[j]-x[i])*y[i+1])/(x[i+1]-x[i])
            inter_x.append(a[j])
            inter_y.append(y_a)
            
    plt.plot(inter_x,inter_y,'-',label="Simple Interpolation")# plotting the simple interpolation
    
    
def cubicSpline(x,y):
    #to find the value of the 2nd order differential at each data point. This is done by generating a series of equations following A.f=b where A is the a tridiagonal matrix (square) of the coefficient of the 2nd order differential, f is a matrix of 2nd order differentials at each point, and b is matrix representing constants based on the data points.  
    A=np.zeros([len(x)-2,len(x)-2])# a tridiagonal matrix of the coefficient of the 2nd order differential 
    b=[]
    counter=1# counter used to position the 2nd order differentials at point x[i] along the diagonal
    for i in range(1,len(x)-1,1): #this uses the natural spline boundary conditions, therefore conditions are used to accound for this in the A matrix
        
        if i==1:
            a_2=(x[i+1]-x[i-1])/3
            a_3=(x[i+1]-x[i])/6 
            A[0][0]=a_2
            A[0][1]=a_3
        elif i>1 and i<len(x)-2:
            a_1=(x[i]-x[i-1])/6 #cofficients in the fundamental equation
            a_2=(x[i+1]-x[i-1])/3
            a_3=(x[i+1]-x[i])/6
            
            A[i-1][counter-1]=a_1
            A[i-1][counter]=a_2
            A[i-1][counter+1]=a_3
            
            counter=counter+1
        else:
            a_1=(x[i]-x[i-1])/6
            a_2=(x[i+1]-x[i-1])/3
            
            A[i-1][counter-1]=a_1
            A[i-1][counter]=a_2
        
    for i in range(1,len(x)-1,1):
        b_1=(y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1])
        b.append(b_1) #right hand side of the fundamental equation 
    
    f_2=LU.solve(A,b)
    
    inbetween=np.append([0],[f_2])
    f_2_corr=np.append([inbetween],[0]) #appends f''_0=f''_n=0 to the arrays of the 2nd order differentials
    
    final_x=[]
    final_y=[]
    for i in range(len(x)-1):
        a= np.linspace(x[i],x[i+1],20,endpoint=True)
        for j in range(len(a)):
            final_x.append(a[j])
            A_x=(x[i+1]-a[j])/(x[i+1]-x[i])#finding the coefficients for the cubic interpolation equations 
            B_x=1-A_x
            C_x=(pow(A_x,3)-A_x)*pow(x[i+1]-x[i],2) /6
            D_x=(pow(B_x,3)-B_x)*pow(x[i+1]-x[i],2) /6
            
            y_new=A_x*y[i]+B_x*y[i+1]+C_x*f_2_corr[i]+D_x*f_2_corr[i+1]#finding the y values for each x point based on the cubic interpolation equation 
            final_y.append(y_new)
    
    plt.plot(final_x,final_y,label="Cubic Spline")#plots the cubic spline
    
    return A,final_x,final_y
            
#%%Testing
x= [-2.1,-1.45,-1.3,-0.2,0.1,0.15,0.9,1.1,1.5,2.8,3.8]
y= [0.012155,0.122151,0.184520,0.960789,0.990050,0.977751,0.422383,0.298197,0.105399,3.936690/10000,5.355348/10000000]
plt.plot(x,y,'.',label="Data")
simpleInter(x,y)
plt.legend()
plt.show()

A=cubicSpline(x,y)
plt.show()    
    

