#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Numpy

import numpy as np


# In[2]:


#Randomly Selecting the variables

p = 8
g = 10
h = 6
sigma = 0.8
np.random.seed(25)
X = np.random.randint(100,size = p)
Q = 8
M = np.random.randint(100, size = Q)
B = np.random.randint(100, size = Q)
var = 0
l=0
n=8
lambda1 = 0.2
lambda2 = 0.3
phai = 0


# In[3]:


# EVALUATING EQUATION 1

for q in range(Q):
    l = M[q]*g*h*sigma*(np.multiply(B[q],X.transpose())+B[q])
var = np.sum(l)
print("Output of Equation 1 is:{}".format(var))


# In[4]:


# EVALUATING EQUATION 2

p1 = 1/sigma
for q in range(2):
    func = ((p1*var-np.multiply(B[q],X.transpose()))*(-0.5))**2
    #func = np.exp(func)   overflow encountered runtime error
    i = M[q]*p*func/(1.414*3.14)
I=np.sum(i)
I = np.log(I)
for i in range(n):
    I+=l
I = (np.sum(I))/n

for q in range(2):
    phai = M[q]*B[q]/sigma
    var1 = lambda1*phai


for j in range(p):
    I+=l
var2 = lambda2*I
var_2 = var1+var2
out_2 = min(var_2)
print("Output of Equation 2 is:{}".format(out_2))


# In[5]:


# EVALUATING EQUATION 3

Z=0
tou = 1000
for j in range(p):
    Z = 1-(np.exp(phai/tou))
var3 = lambda2*Z
var_3 = var1+var3
print("Output of Equation 3 is:{}".format(var_3))


# In[6]:


# EVALUATING EQUATION 4
for p in range(2):
    func = ((p1*var-np.multiply(B[q],X.transpose()))*(-0.5))**2
    var4 = M[q]*p*func/(1.414*3.14)
var_4 = var4/(var3+var2)
print("Output of Equation 4 is:{}".format(var_4))


# In[7]:


# EVALUATING EQUATION 5
for q in range(2):
    func = ((p1*var-np.multiply(B[q],X.transpose()))*(-0.5))**2
    var5 = M[q]*p*func/(3.14)
var51 = lambda1*var5
for i in range(p):
    for q in range(2):
        phai = M[q]*B[q]/sigma
        var52 = lambda1*phai
var_5 = var51+var52
print("Output of Equation 5 is:{}".format(var_5))


# In[8]:


# EVALUATING EQUATION 6
np.random.seed(0)
def compute(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output
        
values = np.random.randint(1, 10, size=5)
out = compute(values)
for p in range(2):
    var6 = np.log(out)
tou = 1000
for j in range(p):
    Z = 1-(np.exp(phai/tou))
    var61 = lambda2*Z
for q in range(2):
    phai = M[q]*B[q]/sigma
    var62 = lambda1*phai
for q in range(Q):
    l = M[q]*g*h*sigma*(np.multiply(B[q],X.transpose())+B[q])
var63 = np.sum(l)
val_6 = var6+var61+var62+var63
print("Output of Equation 6 is:{}".format(val_6))


# In[9]:


# EVALUATING EQUATION 7

for p in range(2):
    func = ((p1*var-np.multiply(B[q],X.transpose()))*(-0.5))**2
    i = M[q]*p*func/(1.414*3.14)
I=np.sum(i)
I = np.log(I)
for i in range(n):
    I+=l
I = (np.sum(I))/n
for q in range(2):
    phai = M[q]*B[q]/sigma
    var1 = lambda1*phai
for q in range(p):
    I+=l
var7 = lambda1*I
out_7 = var1+var7
print("Output of Equation 7 is:{}".format(out_7))


# In[ ]:




