import sys
import math
import random
################
##  py GradientDescent.py datafile trainlabelfile
################

#############
## Sub routines
#############
def dotproduct(x, y):
    if(len(x) == len(y)):
        dp = 0
        for i in range(0, len(x), 1):
            dp += x[i]*y[i]
    return dp

################
##Read Data
################
datafile = sys.argv[1]
f = open(datafile)
data = []
i=0
l=f.readline()
while(l != ''):
    a=l.split()
    l2=[]
    for j in range(0,len(a),1):
        l2.append(float(a[j]))
    l2.append(1)
    data.append(l2)
    l=f.readline()

rows = len(data)
cols = len(data[0])
f.close()

################
##Read Labels
################
labelfile = sys.argv[2]
f = open(labelfile)
trainlabels = {}
n = []
n.append(0)
n.append(0)
l = f.readline()
while(l != ''):
    a = l.split()
    trainlabels[int(a[1])] = int(a[0])
    l = f.readline()
    n[int(a[0])] += 1

############
##Initialize w
############ 
w=[]
for i in range(0,cols):
    w.append(0.002*random.random()-0.001)

############
## Add eta
############
eta = 0.01

############
## Add stop_condition
############
stop_condition = 0.0000001

dellf = []
for j in range(0, cols, 1):
    dellf.append(0)

lastObjective = 0
df1 = 0
############
## Iteration
############
for k in range(0,100000,1):
      
    dellf=[0]*cols
    
    ## Compute dellf ##
    for i in range(0, rows, 1):
        if(trainlabels.get(i) != None):
            dp = dotproduct(w, data[i])
            df1 = (1/(1+math.exp(-dp)))
            for j in range(0, cols, 1):
                dellf[j] = dellf[j] + (trainlabels.get(i)-df1)*data[i][j]

    ## Update w ##
    for j in range(0, cols, 1):
        w[j] = w[j] + (eta * dellf[j])

    ## Calculating error ##
    '''error = 0
    objective = 0
    for i in range(0, rows, 1):
        if(trainlabels.get(i) != None):
            dp = dotproduct(w, data[i])
            gradient = (trainlabels.get(i)-dp)**2
            error = error + gradient'''
            
    objective = error
    
    if(abs(lastObjective - objective) < stop_condition):
        break
    lastObjective = objective
    print("Objective is : ", error)

normw=0
for j in range(0,cols-1,1):
    normw+=w[j]**2

print("eta: ", eta)
print("Stop Condition: ", float(stop_condition))
print("w: ", w)
normw = math.sqrt(normw)
print("||W||=",normw)
normw = math.sqrt(w[0]**2 + w[1]**2)
d_origin = w[cols-1]/normw
print("Distance to origin: ", d_origin)

###########################
## Clasify unlabeled points
###########################

for i in range(0, rows, 1):
    if(trainlabels.get(i) == None):
        dp = dotproduct(w, data[i])
        if (dp > 0):
            print("1 ", i)
        else:
            print("0 ", i)
