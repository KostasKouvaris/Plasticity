################################################################################
############################# IMPORTING ########################################
################################################################################

cimport cython
import numpy as np
cimport numpy as np
from matplotlib import pyplot as plt

#import from C libraries
from libc.math cimport exp, sqrt
from libc.stdlib cimport rand, RAND_MAX

#import from custom Modules
from supFunctions import get_norm_cumsum, generate_training, norm
from globalVariables import DIST_FUNC, NET_STRUCTURE, CON_MUT_RATE, NET_SIZE, ACT_FUN_HIDDEN, ACT_FUN_OUTPUT, MAX_LAYER_SIZE
from constants import EXPERIMENT_CODE, TRAINING_SIZE, TEST_SIZE, TEACHER_FUNCTION, NUM_GENS_PER_ENVIRONMENT, NUM_ENV_PER_GENERATION, IS_BATCH, INCL_BIASES, NET_ACT_FUNCS, POP_SIZE, CROSSOVER_RATE, SEL_STRENGTH, MUT_SIZE, PLASTICITY_COST, INCL_CROSSOVER

################################################################################
############################# CONSTANTS ########################################
################################################################################

DOUBLE_t = np.double

# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False
# cython: nonecheck = False

################################################################################
############################## CLASSES #########################################
################################################################################


############################# INDIVIDUAL #######################################   
    
@cython.freelist(100)
cdef class Individual(object):
    # ATTRIBUTES
    # phen: Adult phenotype (protected)
    # fitness: Fitness (protected)
    # network: Object of class Network (protected)
    # conMutRate: Mutation rate per weight
    # netSize
    
    cdef public:
        bint inclBiases        
        double fitness, conMutRate
        size_t numLayers, netSize
        list weights, bias
        act_hid
        act_out     
    
    # CONSTRUCTOR
    def __init__(Individual self, int[:] structure=NET_STRUCTURE, list act_funcs = NET_ACT_FUNCS, bint inclBiases=INCL_BIASES, conMutRate = CON_MUT_RATE, netSize = NET_SIZE, actFunHidden=ACT_FUN_HIDDEN, actFunOutput=ACT_FUN_OUTPUT,  simple = False):   
        
        if simple:                   
            return  
            
        cdef size_t i            
        
        self.numLayers = <size_t>len(structure) - 1
        self.inclBiases = inclBiases          

        #initialise layers
        self.weights = [None] * self.numLayers
        self.bias = [None] * self.numLayers    
                
        # Add layers according to network structure
        for i in range(self.numLayers):
            self.weights[i] = np.zeros((structure[i], structure[i+1]), dtype = DOUBLE_t)
            self.bias[i] = np.zeros(structure[i+1], dtype = DOUBLE_t)
            
        self.act_hid = actFunHidden
        self.act_out = actFunOutput
        self.conMutRate = conMutRate
        self.netSize = netSize
        self.fitness = 1        
       
    #COPY    
    cpdef public Individual copy(Individual self):
        cdef size_t i
        cdef Individual newInd = Individual(simple=True)
        newInd.inclBiases = self.inclBiases
        newInd.fitness = self.fitness
        newInd.numLayers = self.numLayers  
        newInd.act_hid = self.act_hid
        newInd.act_out = self.act_out    
        newInd.conMutRate = self.conMutRate
        newInd.netSize = self.netSize           
        newInd.weights = [self.weights[i].copy() for i in range(self.numLayers)]
        newInd.bias = [self.bias[i].copy() for i in range(self.numLayers)]
        return newInd        
    
    # METHODS   
    cpdef np.ndarray[double, ndim = 2] develop(Individual self, np.ndarray[double, ndim = 2] envCues):
        # develop: Activation function, generates attribute phen 
        # (phenotype) based on environmental input (envCues)
        # use function get_phenotype to return calculated phenotypes
        # envCues: environmental cues, np.array vector of size equal to 
        # input layer

        cdef size_t i, j, k, l, nRows, nCols
        cdef double sumNet
        cdef np.ndarray[double, ndim = 2] phen      
        cdef int numEnvs = envCues.shape[0]
        cdef int numCues = envCues.shape[1] 

        
        '''
        #initialise phen
        phen = np.empty(shape=(numEnvs, numCues))
         
        for k in range(numEnvs): #foreach environment
            
            #develop adult phenotype for envCues[k,:]
            for l in range(self.numLayers): #foreach layer
                
                #calculate activity for l-layer
                #get size of the respective weight matrix
                nRows = self.weights[l].shape[0]
                nCols = self.weights[l].shape[1] 
                
                for i in range(nRows):
                    sumNet = 0
                    for j in range(nCols):
                        sumNet += self.weights[l][i,j] * (envCues[k,j] if l==0 else phen[k,j]) + self.bias[l][i]
                    phen[k,i] = self.act_out(sumNet) if l==self.numLayers-1 else self.act_hid(sumNet)
                
        return phen
        '''
        
        # calculate activity across layers
        for l in range(self.numLayers): #foreach layer     
            phen = (envCues if l==0 else phen).dot(self.weights[l]) + self.bias[l]   
            phen = self.act_out(phen) if l == self.numLayers-1 else self.act_hid(phen)        
        return phen

    cpdef void mutate(Individual self):
        
        # mutate: mutation function    
        cdef size_t i, numLayers, nRows, nCols
                                                           
        numLayers = self.numLayers-1
        for weights in self.weights:
            nRows = weights.shape[0]
            nCols = weights.shape[1]
            for i in range(nRows):
                for j in range(nCols): 
                    if rand()/float(RAND_MAX) < self.conMutRate:
                        weights[i,j] += np.random.normal(0, MUT_SIZE)
        if INCL_BIASES:
            for bias in self.bias:
                nRows = bias.shape[0]     
                for i in range(nRows):
                    if rand()/float(RAND_MAX) < self.conMutRate:
                        bias[i] += np.random.normal(0, MUT_SIZE)    
         
    cpdef np.ndarray[double, ndim=2] eval_performance(Individual self, np.ndarray[double, ndim=2] envCues, np.ndarray[double, ndim=2] target, double plasticityCost = PLASTICITY_COST):
        # evaluate performance of the organism compared to a set of target phenotypes
        # target: selective environment, vector of size equal to phenotype

        cdef double benefit, cost, performance, sumP
        cdef size_t i, j, nRows, nCols, targetNum, phenSize
        cdef np.ndarray[double, ndim = 2] phen
        
        phen = self.develop(envCues)
        
        #benefit
        benefit = 0
        targetNum = target.shape[0]
        phenSize = target.shape[1]
        for i in range(targetNum): # for each environment
            sumP = 0
            for j in range(phenSize): # for each trait
                sumP += (target[i,j] -  phen[i,j]) ** 2
            benefit += sqrt(sumP)
        benefit /= targetNum
            
        #cost
        cost = 0
        if plasticityCost > 0: #ignore if plasticity is not costly
            nRows = self.weights[0].shape[0]
            nCols = self.weights[0].shape[1]          
            for i in range(nRows):
                for j in range(nCols):
                    cost += self.weights[0][i,j] ** 2
            cost = sqrt(cost/(nRows*nCols))
    
        performance = - benefit - plasticityCost * cost
        return np.asarray(performance)
        
    cpdef void eval_fitness(Individual self, Environment env):
        # evaluate fitness of the organism
        self.fitness = exp(self.eval_performance(env.cues, env.target)/ (2 * SEL_STRENGTH))

    cpdef np.ndarray[double, ndim=2] eval_training_performance(Individual self, Environment env):
        # evaluate performance of the organism across the entire environmental range of the experiment 
        return self.eval_performance(env.trainingInput, env.trainingOutput, plasticityCost = 0)
        
    cpdef np.ndarray[double, ndim=2] eval_test_performance(Individual self, Environment env):
        # evaluate performance of the organism across the entire environmental range of the experiment 
        return self.eval_performance(env.testInput, env.testOutput, plasticityCost = 0)        
        
    cpdef void crossover(Individual self, Individual partner):
        # perform crossover between parents. 
        # crossover is applied on each column, including the corresponding bias
        # node when appropriate.
        pass
        
    # PLOT FUNCTIONS
    cpdef void plot_reaction_norm(Individual self, Environment env, bint showFig = False, bint saveFig = False, str filePath = '', str typeOf = 'fittest'):
        # plot reaction norm
        cdef np.ndarray[double, ndim=2] phen 
        
        fig = plt.figure(figsize=(10,10))
        
        extremIdx = np.array([np.argmax(env.testInput),np.argmin(env.testInput)])
        plt.plot(env.testInput[extremIdx], env.testOutput[extremIdx], 'b--', alpha = 0.5, zorder = 0)
                        
        phen = self.develop(env.trainingInput)
        plt.scatter(env.trainingInput, phen, label = 'Developed phenotype', color = 'orange', marker = 'o')
        
        plt.scatter(env.trainingInput, env.trainingOutput, marker='x', color='red', label = 'Target phenotype')
        
        plt.xlabel('Environmental cue')
        plt.ylabel('Phenotype')
        plt.title('Reaction Norm' + (' - Best Individual' if typeOf == 'best' else ' - Fittest Individual'))
        plt.legend(loc = 'best')
        
        #show figure
        if showFig:
            plt.show()
        
        #save figure
        if saveFig:
            fig.savefig(filePath + typeOf + '_reaction_norm.png')   
                    
            
############################# POPULATION #######################################

cdef class Population(object):
    # ATTRIBUTES
    # ind: List of individuals (protected)
    # popSize: number of individuals
    
    cdef readonly:
        int popSize
        list ind
    
    # CONSTRUCTOR
    def __init__(Population self, int popSize = 1): #default population size is 1. Hill climbing approximation.
        
        self.popSize = popSize
        self.ind = [Individual()] * self.popSize
        
    # METHODS
    cpdef void next_gen(Population self, Environment env):
        
        
        # iterate development/selection/reproduction for one generation
        # develop and evaluate fitness of individuals in current population
        if not self.popSize == 1:
        
            # select best individuals and create new population
            self.ind = self.select_new_pop()
            
            # mutate
            self.mutate()
            
            # evaluate fitness
            self.eval_fitness(env)
        
        else:
            # iterate development and selection
            self.ind[0].eval_fitness(env)
            
            # copy individual to child, mutate develop and eval fitness
            child = self.ind[0].copy()
            
            # mutate
            child.mutate()
            child.eval_fitness(env)

            # If the child is fitter than the parent change population
            if child.fitness > self.ind[0].fitness:
                self.ind[0] = child.copy()
 
    cdef void mutate(Population self):
        # mutate each individual in the population
        for member in self.ind:
          member.mutate()                  
            
    cdef eval_fitness(Population self, Environment env):
        for member in self.ind:
            member.eval_fitness(env)

    cdef list select_new_pop(Population self):
        # select individuals using fitness proportion selection
        cdef np.ndarray[double, ndim = 1] p = self.get_fitness()
        picks = np.random.choice(range(self.popSize), size = self.popSize, p = p/sum(p)) 
        return [self.ind[k].copy() for k in picks]                
            
    ### GET FUNCTIONS
    cdef np.ndarray[double, ndim = 1] get_fitness(Population self):
        # return a list of individuals' fitness
        return np.array([ind.fitness for ind in self.ind])

    cpdef Individual get_best_individual(Population self, Environment env):
        # return the index / position of the best individual in the population
        return self.ind[np.argmax(self.get_train_performance(env))]
        
    cpdef Individual get_fittest_individual(Population self):
        # return the index / position of the best individual in the population
        return self.ind[np.argmax(self.get_fitness())]        

    cpdef double get_top_fitness(Population self):
	# evaluate the best fitness among all individuals
        return np.max(self.get_fitness())
			
    cpdef double get_mean_fitness(Population self):
        # evaluate mean fitness of all individuals
        return np.mean(self.get_fitness())
        
    cpdef list get_performance(Population self, Environment env):
        # return a list of individuals' performance across the whole training set
        return [ind.eval_performance(env.cues, env.target, plasticityCost = 0) for ind in self.ind]

    cpdef double get_top_performance(Population self, Environment env):
	# evaluate the best performance among each individual across the whole training set
        return np.max(self.get_performance(env))
		
    cpdef double get_mean_performance(Population self, Environment env):
	# evaluate mean performance among each individual across the whole training set
        return np.mean(self.get_performance(env))        
		
    cpdef list get_train_performance(Population self, Environment env):
        # return a list of individuals' performance across the whole training set
        return [ind.eval_training_performance(env) for ind in self.ind]

    cpdef double get_top_training_performance(Population self, Environment env):
	# evaluate the best performance among each individual across the whole training set
        return np.max(self.get_train_performance(env))
		
    cpdef double get_mean_training_performance(Population self, Environment env):
	# evaluate mean performance among each individual across the whole training set
        return np.mean(self.get_train_performance(env))
        
    cpdef list get_test_performance(Population self, Environment env):
        # return a list of individuals' performance across the whole training set
        return [ind.eval_test_performance(env) for ind in self.ind]

    cpdef double get_top_test_performance(Population self, Environment env):
	# evaluate the best performance among each individual across the whole training set
        return np.max(self.get_test_performance(env))
		
    cpdef double get_mean_test_performance(Population self, Environment env):
	# evaluate mean performance among each individual across the whole training set
        return np.mean(self.get_test_performance(env))        

    # PLOT FUNCTIONS
    cpdef void plot_reaction_norms(Population self, Environment env, bint showFig = False, bint saveFig = False, str filePath = ''):
        # plot reaction norm
        cdef np.ndarray[double, ndim=2] phen 
        
        fig = plt.figure(figsize=(10,10))
        
        #plot teacher function
        extremIdx = np.array([np.argmax(env.testInput),np.argmin(env.testInput)])
        plt.plot(env.testInput[extremIdx], env.testOutput[extremIdx], 'b--', alpha = 0.5, zorder = 10, label = 'Maximally-fit phenotype')
        
        if EXPERIMENT_CODE != 'G':                
            #plot individual reaction norms
            for i in range(self.popSize):
                phen = self.ind[i].develop(env.testInput[extremIdx])
                plt.plot(env.testInput[extremIdx], phen, color = '0.75', label = 'Evolved reaction norms' if i==0 else '', alpha = 0.5, zorder = 0)
        else:
            phen = self.ind[0].develop(env.testInput)
            plt.scatter(env.testInput, phen, color = '0.75', label = 'Evolved reaction norms', zorder = 0) 
        
        #plot past selective environments
        plt.scatter(env.trainingInput, env.trainingOutput, marker='x', color='red', label = 'Past phenotypic targets', zorder = 20)
        
        #plot best reaction norm wrt environmental structure
        phen = self.get_best_individual(env).develop(env.trainingInput)
        plt.scatter(env.trainingInput, phen, marker='o', color='green', label = 'Fittest in the past environments', zorder = 15)

        if EXPERIMENT_CODE != 'G':                        
            #plot fittest reaction norm to the current environment
            phen = self.get_fittest_individual().develop(env.trainingInput)
            plt.scatter(env.trainingInput, phen, marker='o', color='orange', label = 'Fittest in the current environment(s)', zorder = 14) 
        
        plt.xlabel('Environmental cue')
        plt.ylabel('Phenotype')
        plt.title('Reaction Norms')
        plt.legend(loc = 'best')
        
        #show figure
        if showFig:
            plt.show()
        
        #save figure
        if saveFig:
            fig.savefig(filePath + '_reaction_norm.png')         

############################# ENVIRONMENT ######################################                     

cdef class Environment(object):
    
    cdef readonly:
        np.ndarray trainingInput, trainingOutput, testInput, testOutput, cues, target
        
    cdef public:
        int numGensPerEnvironment, numEnvPerGeneration, trainingSize, testSize
        bint isBatch
    
    # CONSTRUCTOR
    def __init__(Environment self, trainingSize = TRAINING_SIZE, testSize = TEST_SIZE):
        # numGensPerEnvironment: number of generations per environment
	# numEnvPerGeneration: number of environment per each generation
        self.trainingInput, self.trainingOutput = generate_training(trainingSize, TEACHER_FUNCTION)
        self.testInput, self.testOutput = generate_training(testSize, TEACHER_FUNCTION)
        self.cues, self.target = self.trainingInput, self.trainingOutput
        self.numGensPerEnvironment = NUM_GENS_PER_ENVIRONMENT
        self.numEnvPerGeneration = NUM_ENV_PER_GENERATION
        self.trainingSize = trainingSize
        self.testSize = testSize
        self.isBatch = IS_BATCH
    
    # METHODS
    cpdef void generate_environment(Environment self, int gener):
        # generate a new environment: draw a new sample from the teacher distribution
        # gener: number of current generation    
        
        cdef np.ndarray[int, ndim=1] choices  
        if not self.isBatch and gener % self.numGensPerEnvironment == 0:
            choices = np.random.choice(self.trainingSize, self.numEnvPerGeneration, replace=False)
            self.cues, self.target = self.trainingInput[choices,], self.trainingOutput[choices,]
            
    cpdef void add_noise_out(Environment self, double noiseLevel):
        outputSize = self.trainingOutput.shape[1]
        self.trainingOutput += np.random.normal(0,noiseLevel, size =(self.trainingSize,outputSize))
        