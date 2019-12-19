import numpy as np
#%%
def LUdcomp(a, det=True): #Note: unlike the formulas given in the notes, 
    N=len(a) #dimension of matrices
    L=np.zeros([N,N])# empty (zeros) NxN Lower matrix
    U=np.zeros([N,N])#empty (zeros) NxN Upper matrix
    for j in range(0,N):#loops over the range of jth terms from j=0,1...N-1.
        L[j][j]=1 # fills in L_11,L_22,...L_NN=1
        
        for i in range(0,j+1): #loops over values of i from i=0,1...j
            sum0=0
            for k in range(0,i): #loops over k from k=0,1...i-1
                sum0=sum0+L[i][k]*U[k][j] 
            U[i][j]=a[i][j]-sum0 # equation to find the upper components
        
        for i in range(j+1,N):#loops over values of i from i=j+1,j+2...N-1
            sum1=0
            for k in range(0,j):#loops over k from k=0,1...j-1
                sum1=sum1+L[i][k]*U[k][j]
            L[i][j]=(a[i][j]-sum1)/U[j][j] # equation to find the upper components
            
    if det==True:
        detA=1
        for i in range(N):# the determinant is the product of the diagonal elements of the upper matrix
            detA=detA*U[i][i]
    else:
        detA=None
            
                  
    return L, U, detA

def combine(L,U):
    N=len(L) #dimension of matrices
    C=np.zeros([N,N])
    for i in range(N): #adds all elements of the upper and lower matrices, expect for the diagonal.
        for j in range(N):
            if i != j:   
                C[i][j]=L[i][j]+U[i][j]
            else:
                C[i][j]=U[i][j]# for the diagonal elements, the combined matrix just takes the value of the upper matrix.
    return C  


def solve(A,b): #L.U.x=b, set b as an Nx1 matrix(/array) 
    L,U, det=LUdcomp(A, det=False)
    N=len(L) #dimension of matrices
    x=np.zeros((N)) #this array will the final solution
    y=np.zeros((N)) #this array is an intermidate solution
    
    y[0]=b[0]/L[0][0] #setting the first entry for the intermidate solution(y)
    
    
    
    for i in range(0,N): #the lower matrix acts on the (dummy) intermidate solution (y) such that L.y=b. y is solved for by using forwards subsitution  
        sum0=0
        for j in range(0,i):
            sum0=sum0+L[i][j]*y[j]
        y[i]=(b[i]-sum0)/L[i][i]
    
    x[N-1]=y[N-1]/U[N-1][N-1] #setting the last entry for the actual solution(x)
    #print(x[N-1])
    
    for i in range(N-1,-1,-1): # the upper matrix acts on the acutal solution (x) such that U.x=y. x is solved for by using backwards subsitution. 
        sum0=0
        for j in range(i+1,N):
            sum0=sum0+U[i][j]*x[j]
        x[i]=(y[i]-sum0)/U[i][i]
        
    
    return x


def invM(A):
    
    L,U,det=LUdcomp(A, det=False)
    N=len(L) #dimension of matrices
    I_1=[] 
    A_1=np.zeros([N,N]) #the matrix that will be the inverse matrix 
    
    for i in range(0,N): # NxN identity matrix seperated into columns(I_1)
        a=np.zeros((N))
        a[i]=1
        I_1.append(a)
    
    for j in range(len(A_1)): #finding the inverse matrix by using the previous "solve" function(but now the b input is the column of idenity matrix)
        b=solve(A,I_1[j])
        for k in range(len(I_1[j])):
            A_1[k][j]=b[k] #the solutions are combined into a NxN matrix, which is the inverse matrix.
    
    return A_1
    
    
            
    
      
#%% Testing
eg1=[[3,1,0,0,0],
   [3,9,4,0,0,],
   [0,9,20,10,0],
   [0,0,-22,31,-25],
   [0,0,0,-55,61]] #This is the input NxN matrix

eg2=[[-4,1],
     [15,-5]]

L,U,detA=LUdcomp(eg1)
C=combine(L,U)
print("U=",U)
print("L=",L)  
#print("C=",C)
#print(detA)

beg1=[2,5,-4,8,9]
beg2=[-2,5]
solv=solve(eg1,beg1)
print(solv)
invMatrix=invM(eg2)
print(np.round(np.dot(invMatrix,eg2)))# rounding to ignore the rounding-errors
