################################################################################
############################# IMPORTING ########################################
################################################################################

import os
import shelve
import numpy as np
cimport numpy as np
import constants as const
import pickle
from numpy cimport ndarray
from supFunctions import plot_performance
from classes import Population, Environment

################################################################################
############################## EXPERIMENTS #####################################
################################################################################

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def load(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def save_global_vars_txt(path):
    tmpDir = [item for item in dir(const) if not item.startswith("__") and item.isupper()]
    with open(path + 'global_vars.txt', 'w') as output: 
        for key in tmpDir:
            try:
                output.write(key + ': ' + str(getattr(const,key)) + '\n')  
            except TypeError:
                print('ERROR writing')                

def save_global_vars(path):
    # save global variables
    
    #using pickles
    #tmpDir = [item for item in dir(const) if not item.startswith("__") and item.isupper()]

    #with open(filename, 'wb') as output:     # 'wb' instead 'w' for binary file
    #    for key in tmpDir:
    #        print key
    #        pickle.dump(getattr(const,key), output, pickle.HIGHEST_PROTOCOL)

    #using shelve
    tmpShelf = shelve.open(path + 'global_vars.out','n') # 'n' for new    
    tmpDir = [item for item in dir(const) if not item.startswith("__") and item.isupper()]
    for key in tmpDir:
        try:
            tmpShelf[key] = getattr(const, key)
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    tmpShelf.close()                  

def load_global_vars(filename):
    # load global variables
    
    #using shelve
    tmpShelf = shelve.open(filename) 
    for key in tmpShelf:
        try:
            setattr(const,key,tmpShelf[key])
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    tmpShelf.close()  

def run_instance(str fileDir):
    run_instance_cy(fileDir)

cdef void run_instance_cy(str fileDir):

    #run a simulation instance

    cdef size_t gen_t, i, saveInterval, gen_idx, initGenCount
    cdef np.ndarray[double, ndim = 1] p 
    cdef np.ndarray[double, ndim=2] performance, initialPerformance
    cdef size_t popSize = const.POP_SIZE
    cdef size_t generations = const.GENERATIONS
    cdef bint isBatch = const.IS_BATCH
    cdef bint inclTestPlot = const.INCL_TEST_PLOT    
    cdef double noiseLevel = const.NOISE_LEVEL
    cdef str experimentCode = const.EXPERIMENT_CODE
    print(const.POP_SIZE)
    #variable validation
    assert(noiseLevel == 0), "Environment is noisy"
    assert(inclTestPlot == False), "Including test plot"
    if experimentCode in ['A', 'C', 'D', 'F']:
        assert(isBatch == False), "Attempted to run in Batch mode"
    elif experimentCode in ['B', 'E', 'G']:
        assert(isBatch == True), "Attempted to run in Online mode"
    if experimentCode in ['A', 'B', 'C']:
        assert(popSize != 1), "Population size should be 1"
    elif experimentCode in ['D', 'E', 'F', 'G']:
        assert(popSize == 1), "Population size should be greater than 1"
    
    #initialisation                     
    POP = Population(popSize)
    env = Environment()
    env.add_noise_out(noiseLevel)
    
    if not isBatch:
        saveInterval = env.numGensPerEnvironment
        if int(generations/env.numGensPerEnvironment) > 1000:
            saveInterval = int(int(generations/env.numGensPerEnvironment)/1000) * env.numGensPerEnvironment
    else:
        saveInterval = max(1,int(generations/1000))
    performance = np.empty(shape = (int(generations/saveInterval) + 1, 1 + ((1 if isBatch else 2) + (1 if inclTestPlot else 0)) * (1 if POP.popSize == 1 else 2)))

    #store performance at time step 0
    #initialise
    tempArray = []
    #store generation
    tempArray.append(0)                      
    #top performance
    #store top performance on current environment(s). in batch learning this is equivalent to the training performance
    tempArray.append(POP.get_top_performance(env))
    #store top training performance. skip in batch learning
    if not isBatch:
        tempArray.append(POP.get_top_training_performance(env))       
    #store top test performance
    if inclTestPlot:
        tempArray.append(POP.get_top_test_performance(env))       
    #mean performance
    if not POP.popSize == 1:
        #store mean performance on current environment(s). in batch learning this is equivalent to the training performance
        tempArray.append(POP.get_mean_performance(env))            
        #store mean training performance. skip in batch learning
        if not isBatch:
            tempArray.append(POP.get_mean_training_performance(env))
        #store mean test performance
        if inclTestPlot:
            tempArray.append(POP.get_top_mean_performance(env))                
    #store values to performance variable
    performance[0,:] = np.array(tempArray)    

    if experimentCode == 'A':
        
        initGenCount = 10 * env.numGensPerEnvironment
        print(initGenCount, generations)
        assert(initGenCount <= generations), "Initial generations smaller than total generations"
        initialPerformance = np.empty(shape = (initGenCount + 1,5))
        initialPerformance[0,:] = np.array(tempArray) #np.array([tempArray[i] for i in [0,1,3]])    
    elif experimentCode == 'B':     
        initGenCount = 3000
        assert(initGenCount <= generations), "Initial generations smaller than total generations"
        initialPerformance = np.empty(shape = (initGenCount + 1,3))
        initialPerformance[0,:] = np.array(tempArray)
    elif experimentCode == 'D':
        initGenCount = 10 * env.numGensPerEnvironment
        assert(initGenCount <= generations), "Initial generations smaller than total generations"
        initialPerformance = np.empty(shape = (initGenCount + 1,3))
        initialPerformance[0,:] = np.array(tempArray) #np.array([tempArray[i] for i in [0,1,3]])    
    elif experimentCode == 'E':     
        initGenCount = 3000
        assert(initGenCount <= generations), "Initial generations smaller than total generations"
        initialPerformance = np.empty(shape = (initGenCount + 1,2))
        initialPerformance[0,:] = np.array(tempArray)              
    elif experimentCode == 'T':
        initGenCount = generations/10
        initialPerformance = np.empty(shape = (initGenCount + 1,2))
        initialPerformance[0,:] = np.array(tempArray)
        
    print(POP.popSize)
            
    #main
    for gen_t in range(generations):
    
        # print progress (%)
        if 100 * float(gen_t) / generations % 2 == 0:
            print("%.2f"% (100*float(gen_t)/generations) + "%")        
                  
        # change environment
        env.generate_environment(gen_t)
    
        # next generation        
        POP.next_gen(env)                                         

        # store initial performance        
        if experimentCode == 'A':
            if gen_t < initGenCount:
                initialPerformance[gen_t + 1,:] = np.array([gen_t+1,POP.get_top_performance(env), POP.get_top_training_performance(env), POP.get_mean_performance(env), POP.get_mean_training_performance(env)])
        elif experimentCode == 'B':
            if gen_t < initGenCount:
                initialPerformance[gen_t + 1,:] = np.array([gen_t+1,POP.get_top_performance(env), POP.get_mean_performance(env)])
        elif experimentCode == 'D':
            if gen_t < initGenCount:
                initialPerformance[gen_t + 1,:] = np.array([gen_t+1,POP.get_top_performance(env), POP.get_top_training_performance(env)])
        elif experimentCode == 'E':
            if gen_t < initGenCount:
                initialPerformance[gen_t + 1,:] = np.array([gen_t+1,POP.get_top_performance(env)])
        elif experimentCode == 'T':
            if gen_t < initGenCount:
                initialPerformance[gen_t + 1,:] = np.array([gen_t+1,POP.get_top_performance(env)])

        # store performance evaluation       
        if (gen_t+1) % saveInterval == 0:
            
            #initialise
            tempArray = []
            
            #get index
            gen_idx = int(gen_t/saveInterval) + 1
 
            #store generation
            tempArray.append(gen_t)                      
 
            #top performance
            #store top performance on current environment(s). in batch learning this is equivalent to the training performance
            tempArray.append(POP.get_top_performance(env))
            
            #store top training performance. skip in batch learning
            if not isBatch:
                tempArray.append(POP.get_top_training_performance(env))
                
            #store top test performance
            if inclTestPlot:
                tempArray.append(POP.get_top_test_performance(env))
            
                
            #mean performance
            if not POP.popSize == 1:
                #store mean performance on current environment(s). in batch learning this is equivalent to the training performance
                tempArray.append(POP.get_mean_performance(env))
                            
                #store mean training performance. skip in batch learning
                if not isBatch:
                    tempArray.append(POP.get_mean_training_performance(env))
                
                #store mean test performance
                if inclTestPlot:
                    tempArray.append(POP.get_top_mean_performance(env))                
            
            
            #store values to performance variable
            performance[gen_idx,:] = np.array(tempArray)
       
    #save data
    #generate directory
    path = os.path.abspath(fileDir) + '/'
    print(path)
    ensure_dir(path)
    
    #store simulation specifications - global variables
    save_global_vars(path)
    
    #store simulation specifications - global variables - txt file
    save_global_vars_txt(path)
    
    #store population
    #save_object(POP, path + 'population.p')
        
    #store and show figures
    plot_performance(performance,experimentCode, filePath=path,saveData = True, showFig = True, saveFig = True, initData = initialPerformance if experimentCode != 'G' else None)
    POP.plot_reaction_norms(env, filePath=path, showFig = True, saveFig = True)
    #POP.get_best_individual(env).plot_reaction_norm(env, filePath=path, showFig = True, saveFig = True, typeOf = 'best')
    #POP.get_fittest_individual().plot_reaction_norm(env, filePath=path, showFig = True, saveFig = True, typeOf = 'fittest')
    