################################################################################
############################# IMPORTING ########################################
################################################################################

# cython: profile=True

cimport cython
import pickle
import matplotlib.pyplot as plt
from numpy cimport ndarray
from numpy import dot, ndarray, sqrt, sum, asarray, tanh, cumsum, random
from libc.math cimport exp, sqrt
import numpy as np
cimport numpy as np
from constants import POP_SIZE, IS_BATCH, INCL_TEST_PLOT

################################################################################
######################### SUPPORT FUNCTIONS ####################################
################################################################################

#####assa@cython.boundscheck(False) # turn off bounds-checking for entire function
#  as@cython.wraparound(False)  # turn off negative index wrapping for entire function

cdef ndarray[double, ndim=2] identityFunction_cy(ndarray[double, ndim=2] x):
    # identity function
    return x
 
cdef ndarray[double, ndim=2] logistic_cy(ndarray[double, ndim=2] x):
    # logistic function for np arrays
    return 1/(1+exp(-x))           

cpdef double dist(ndarray[double, ndim=2] f, ndarray[double, ndim=2] g, str distType):
    # returns the distance between x and y using distType measure
    #MEAN
    cdef double d
    if distType == 'euclidean':
        d = euclidean_func_cy(f,g) #np.sqrt(np.sum((f - g) ** 2, axis=1)).mean()
    elif distType == 'dotProd':
        d = - dot(f.T, g).mean()
    return d
      
cdef double euclidean_func_cy(ndarray[double, ndim=2] f, ndarray[double, ndim=2] g):
    return sqrt(sum((f - g) ** 2, axis=1)).mean()    

def identityFunction(x):
    # identity function
    return x
    
def logistic(x):
    # logistic function for np arrays
    return 1/(1+np.exp(-x))    

def get_dist_func(distType):
    dFunc = None
    if distType == 'euclidean':
        dFunc = euclidean_func
    return dFunc
  
def euclidean_func(f, g):
    return sqrt(sum((f - g) ** 2, axis=1)).mean()

def calc_mut_rate(netSize, indMutRate):
    # returns the rate of mutation per link given network size and the mutation
    # rate for each individual
    return (1.0/netSize) * indMutRate
        
def calc_net_size(netStructure, inclBiases):
    # returns the total number of connections for a given a network structure
    # netStructure: vector with one element per network layer from input to 
    # output, must contain integers. Each number defines the size of the layer, 
    # can be list of array, minimum length 2.
    # inclBiases: boolean, determines if the networks have bias terms as well as 
    # links
    netStructure = asarray(netStructure)
    return sum(netStructure[:-1] * netStructure[1:]) + ( sum(netStructure[1:]) if inclBiases else 0 )
    
def calc_net_probabilities(netStructure, inclBiases):
    pWeights = [a*b for a,b in zip(netStructure[:-1],netStructure[1:])]
    if inclBiases:
        pWeights.extend(netStructure[1:])
    return [x/float(sum(pWeights)) for x in pWeights]
    
def get_norm_cumsum(inputArray):
    #inputArray is 1D array
    return cumsum(inputArray/float(sum(inputArray)))    
    
def get_act_func(name):
    #returns the activation function of the respective name
    if name == 'tanh':
        func = tanh
    elif name == 'identity':
        func = identityFunction
    elif name == 'logistic':
        func = logistic
    return func
    
cpdef double norm(np.ndarray[double, ndim=2] weights, int p):
    cdef double cost = 0
    cdef size_t nRows, nCols, i, j
    nRows = weights.shape[0]
    nCols = weights.shape[1]          
    for i in range(nRows):
        for j in range(nCols):
            cost += weights[i,j] ** p
    return (cost/(nRows*nCols)) ** (1/p)
    
def generate_training(trainSize, teacherFunction):
    # generate training set of size trainSize using the teacherFunction
    # returns two matrices (np arrays) for the inputs and the outputs 
    # respectively
    # each row corresponds to a training sample.
    if teacherFunction == 'linear':
        # generate trainSize points drawn uniformly from [a,b]
        #x1,x2 = 0,10
        #inputArray = random.uniform(low=x1, high=x2, size=(trainSize,1)) 

        # generate trainSize points drawn normal distribution N(1,sigma)
        sigma = 1
        inputArray = random.normal(1,sigma, size = (trainSize,1))

        # apply linear transformation of slope a and intercept b
        a,b = -2,6  #-2,4
        outputArray = a * inputArray + b
    return inputArray, outputArray
    	
def plot_performance(performData, experimentCode, showFig = False, saveFig = False, saveData = False, filePath = '', initData = None):
    
    fig = plt.figure(figsize=(10,10))
    
    plt.rcParams.update({'font.size': 18})
    plt.rc('xtick', labelsize=12) 
    plt.rc('ytick', labelsize=12) 
      
    labels = []
    colors = []
    alphas = []
    linestyles = []
        
    #get X values, environmental cues
    xVals = performData[:,0]
    
    #plot top performance
    #top performance on current environment(s)
    colors.append('green')
    alphas.append(1)
    labels.append('Current and Past Enviroments' if IS_BATCH else 'Current Environments')
    linestyles.append('-')
    
    #top training performance
    if not IS_BATCH:
        colors.append('blue')
        alphas.append(1)
        labels.append('Past Enviroments')
        linestyles.append('-')
               
    #store top test performance
    if INCL_TEST_PLOT:
        colors.append('red')
        alphas.append(1)
        labels.append('Previously Unseen Environments')
        linestyles.append('-')
    
    #plot mean performance    
    if not POP_SIZE == 1:
        labels = [lbl + ' (top)' for lbl in labels]
        
        #mean performance on current environment(s)
        colors.append('green')
        alphas.append(0.5)
        labels.append('Current and Past Enviroments (mean)' if IS_BATCH else 'Current Environments (mean)')
        linestyles.append('--')        
                            
        #store mean training performance. skip in batch learning
        if not IS_BATCH:
            colors.append('blue')
            alphas.append(0.5)
            labels.append('Past Enviroments (mean)')
            linestyles.append('--')
                
        #store mean test performance
        if INCL_TEST_PLOT:
            colors.append('red')
            alphas.append(0.5)
            labels.append('Previously Unseen Environments (mean)')
            linestyles.append('--')
                                 
    for i in range(performData.shape[1]-1):
        plt.plot(xVals, performData[:,i+1], color = colors[i], label = labels[i], linestyle = linestyles[i], alpha = alphas[i])

    plt.legend(loc = 'best')
    plt.xlabel('Generations')
    plt.ylabel('Fit to Environment')  
    plt.title('Population performance over evolutionary time')

    if experimentCode in ['A','B','C','D','E','F']:
        plt.ylim([-12, 0.2])
        plt.legend(loc = 'lower right')              
    
    if experimentCode == 'A':
        # this is an inset axes over the main axes
        a = plt.axes([0.5, .34, .35, .25], facecolor='lightgoldenrodyellow')
        a.plot(initData[:,0], initData[:,1], 'g')
        a.plot(initData[:,0], initData[:,3], 'g--', alpha = 0.5)  
        a.plot(initData[:,0], initData[:,2], 'b')
        a.plot(initData[:,0], initData[:,4], 'b--', alpha = 0.5)           
    elif experimentCode == 'B':
        # this is an inset axes over the main axes
        a = plt.axes([0.5, 0.34, .35, .25], facecolor='lightgoldenrodyellow')
        a.plot(initData[:,0], initData[:,1], 'g')
        a.plot(initData[:,0], initData[:,2], 'g--', alpha = 0.5)        
    elif experimentCode == 'D':
        # this is an inset axes over the main axes
        a = plt.axes([0.5, .34, .35, .25], facecolor='lightgoldenrodyellow')
        a.plot(initData[:,0], initData[:,1], 'g') 
        a.plot(initData[:,0], initData[:,2], 'b')      
    elif experimentCode == 'E':
        # this is an inset axes over the main axes
        a = plt.axes([0.5, 0.34, .35, .25], facecolor='lightgoldenrodyellow')
        a.plot(initData[:,0], initData[:,1], 'g')       
    
    #show figure
    if showFig:
        plt.show()
        
    #save figure
    if saveFig:
        fig.savefig(filePath + 'performance.png')
        
    #save data
    if saveData:
        with open(filePath + 'performance.p', 'wb') as output:
            pickle.dump(performData, output, pickle.HIGHEST_PROTOCOL)
        
