import numpy as np
import random as ran
import time

t=time.process_time()

#EXTREMELY SLOW!!!! (compared to 58seconds of Matlab)
#COMPUTING TIME OF 324.328125 SECONDS...
#0.0016274145153445114 as the final cost

"""DATA PORTION"""

x1=np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7]])
x2=np.array([[0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]])
x1=x1.T
x2=x2.T

size=x1.shape[0]
size_half=int(size/2)
#the argument axis=1 concatenates via column
row1=np.concatenate((np.ones((1,size_half)),np.zeros((1,size_half))),axis=1)
row2=np.concatenate((np.zeros((1,size_half)),np.ones((1,size_half))),axis=1)
y=np.concatenate((row1,row2)) 
y=y.astype(int) #make the matrix into an integer matrix

"""--------------------------------------------------------------------------"""

"""INITAL WEIGHTS AND BIASES"""
np.random.seed(5)
w2=0.5*np.random.randn(2,2)
w3=0.5*np.random.randn(3,2)
w4=0.5*np.random.randn(2,3)
b2=0.5*np.random.randn(2,1)
b3=0.5*np.random.randn(3,1)
b4=0.5*np.random.randn(2,1)

"""--------------------------------------------------------------------------"""

def activate(x,w,b):
    '''x is the input vector, w contains the weights, b contains the shifts
    outputs the output vector'''
    return 1/(1+np.exp(-(np.matmul(w,x)+b))) #matmul for the usual matrix multiplication

def cost(w2,w3,w4,b2,b3,b4):
    costvec=np.zeros((10,1))
    for i in range(10):
        x=np.concatenate((np.array([[x1[i][0]]]),np.array([[x2[i][0]]])),axis=1)
        x=x.T
        a2=activate(x,w2,b2)
        a3=activate(a2,w3,b3)
        a4=activate(a3,w4,b4)
        costvec[i][0]=np.linalg.norm(y[:,i].reshape((y[:,i].size,1))-a4)
    costval=np.linalg.norm(costvec)**2
    return costval

"""FORWARD AND BACKWARD PROPOGATION"""
eta=0.05 #step size(learning rate)
num_iter=int(1e6) #number of SG iterations
savecost=np.zeros((num_iter,1)) #value of cost function at each iteration
for i in range(num_iter):
    k=ran.randint(0,9)
    x=np.concatenate((np.array([[x1[k][0]]]),np.array([[x2[k][0]]])),axis=1)
    x=x.T

    a2=activate(x,w2,b2)
    a3=activate(a2,w3,b3)
    a4=activate(a3,w4,b4)
    #multiply for the Hadamard product of matrices
    delta4=np.multiply(a4,np.multiply(1-a4,a4-y[:,k].reshape((y[:,k].size,1))))
    delta3=np.multiply(a3,np.multiply(1-a3,np.matmul(w4.T,delta4)))
    delta2=np.multiply(a2,np.multiply(1-a2,np.matmul(w3.T,delta3)))

    w2=w2-eta*np.matmul(delta2,x.T)
    w3=w3-eta*np.matmul(delta3,a2.T)
    w4=w4-eta*np.matmul(delta4,a3.T)
    b2=b2-eta*delta2
    b3=b3-eta*delta3
    b4=b4-eta*delta4

    newcost=cost(w2,w3,w4,b2,b3,b4)
    #print(newcost)
    savecost[i][0]=newcost

print(time.process_time()-t)
