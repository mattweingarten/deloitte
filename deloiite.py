
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import math  
##SET up correlation matrix:
##      CH EU US
## CH   1 0.6 0.3
## EU   0.6 1 0.4
## US   0.3 0.4 1
region_sigma = np.genfromtxt("./data/Correlation.csv",delimiter=',',skip_header=1)[: , 1:4]
region_mu = [1,1,1]

##SET up factors matrix:
#aplha_ch, alpha_eu, alpha_us

alphas = np.genfromtxt("./data/Factor_loadings.csv",delimiter=',', skip_header=1)[:,1:4]


##SET mapping loan type to rating
ratvalues = np.genfromtxt("./data/PD_Table.csv",delimiter=',',skip_header=1)[:,1:2]
for i in range(ratvalues.size):
    ratvalues[i] = norm.ppf(ratvalues[i])

def rat2index(str):
    if (str == b'AAA'):
        return 0
    elif(str==b'AA'): 
        return 1
    elif(str==b'A'):
        return 2
    elif(str==b'BBB'): 
        return 3
    elif(str==b'BB'):
        return 4
    elif(str==b'B'):
        return 5
    elif(str==b'CCC'):
        return 6
    else:
       return 7


##CONVERTERS
#============
#We assume data integrity


##Id converters(strip ABC)
id2int = lambda x : int(x[3 :])

## Region converters : CH to 0, EU to 1, US to 2
reg2int = lambda x : 0 if (x == b'CH') else (1 if (x == b'EU') else 2)

##Rating converters : indexes into values from PD_table csv according to strings: AAA, AA, A, BBB, BB, B, CCC, or default, 

rat2float = lambda x : ratvalues[rat2index(x)]


##=============
##END


##Loan, Region,Rating(proabability of default),EAD(loan amount), LGD(Loss given default)

portfolio = np.genfromtxt("./data/Portfolio.csv", delimiter=',',skip_header=1,converters={0: id2int,1: reg2int, 2: rat2float},dtype=[('id', int),('region',int),('thresh',float),('amount',float),('ldg',float)])
## size checks seem to be correct ( size = 38401, consistent with csv file)
assert(portfolio.size == alphas.size/3), "size mismatch"

## sets up gammas from alphas
gammas = np.zeros(portfolio.size)
for i in range(portfolio.size):
     gammas[i] = math.sqrt((1-math.pow(alphas[i].sum(), 2)))

#this function runs 1 montecarlo simulation for all loeans and return the loss
def simulation():
    res = 0
    #random sample for regional covariance
    regsample = np.random.multivariate_normal(region_mu,region_sigma,1)[0]
    #random sample point for each loan
    loansample =  np.random.normal(0,1,portfolio.size)
    for i in range(loansample.size):
        value = np.dot(regsample, alphas[i]) + gammas[i] * loansample[i]
        if(value <= portfolio[i]['thresh']): res += portfolio[i]['ldg'] * portfolio[i]['amount']
    return res
 

##runs simulation 100 000 times
def montecarlo():
    n = 100
    sims = np.zeros(n)
    for i in range(n):
        sims[i] = simulation()
    return sims

#displays results ins histogram
def displayhisto(sims, mean, std):
    n = sims.size
    max = mean +  500000
    min = mean -  500000
    histo, bins = np.histogram(sims,100,(min,max))
    plt.hist(sims,bins)
    plt.show()

##returns mean and standard deviation of normal estimation from data
def describemonte(sims):
    n = sims.size
    std = 0
    mean = np.average(sims)
    for i in range(n):
        std += math.pow((sims[i] - mean), 2)
    return (mean, std/n)   


sims = montecarlo()
mu, std = describemonte(sims)
displayhisto(sims, mu, std)

print((mu,std))