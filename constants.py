################################################################################
############################# CONSTANTS ########################################
################################################################################

# NET_STRUCTURE: vector with one element per network layer from input to output, 
# must contain integers. Each number defines the size of the layer, can be list 
# of array, minimum length 2.
NET_STRUCTURE = [1,1] # Alternative configurations for high complexity: [1,10,6,25,5,1], [1,10,10,5,1], [1,2,5,10,5,2,1]
NET_ACT_FUNCS = ['tanh','linear']

# INCL_BIASES: boolean, determines if the networks have bias terms 
INCL_BIASES = True

# ACT_FUN_HIDDEN: activation function for hidden layers
ACT_FUN_HIDDEN = 'tanh'

# ACT_FUN_OUTPUT: activation function for output layer
ACT_FUN_OUTPUT = 'identity'

# DISTANCE_METRIC: function used to calculate distance between phenotype and 
# target 
DISTANCE_METRIC= 'euclidean'

# SEL_STRENGTH: selection strength. The lower the value, the higher the strength of selection is.
SEL_STRENGTH = 0.2 # Very strong selection: 0.005

# IND_MUT_RATE: mutation rate per capita
IND_MUT_RATE = 1

# MUT_SIZE: change in weight per mutation (std: normal distribution)
MUT_SIZE = 0.001 # Fast mutation: 0.01, Slow mutation: 0.0001

# INCL_CROSSOVER: boolean, determines whether crossover is taken into account
INCL_CROSSOVER = False

# CROSSOVER_RATE: proportion of weights and biases inhertied from partner
CROSSOVER_RATE = 1

# PLASTICITY_COST: cost of plasticity
# penalty for individuals with plastic responses, applied to fitness calculations
PLASTICITY_COST = 0.1
 
# TRAINING_SIZE: number of samples in the training set
TRAINING_SIZE = 10

# TEST_SIZE: number of samples in the test set
TEST_SIZE = 200

# POP_SIZE: number of individuals in population
POP_SIZE = 1 # 1 for Hill-Climber simulation, 1000 for population based simulations

# NUM_GENS_PER_ENVIRONMENT: number of generations per environmental changes
NUM_GENS_PER_ENVIRONMENT = 40000 # 4000 for population based simulation, 40000 for HC

# NUM_ENV_PER_GENERATION: number of environments per generation
NUM_ENV_PER_GENERATION = 1

# GENERATIONS: number of generations
GENERATIONS = 2*10000000 

# IS_BATCH: boolean, determines if fitting with batch (True) or not (False)
# if True, no need to update training set every time
IS_BATCH = False

# TEACHER_FUNCTION: the function that describes the underlying problem structure
TEACHER_FUNCTION = 'linear'

#ELITIST_PROPORTION: the proportion of the population retained by the elitist selection
ELITIST_PROPORTION = 0.1

#LAMBDA: Average number of mutations per individual
LAMBDA = 0.2

#NOISE_LEVEL: Level of stochastic noise on the training set
NOISE_LEVEL = 0 #0.1

#INCL_TEST_PLOT:
INCL_TEST_PLOT = False

#EXPERIMENT_CODE
#A: Population, Coarse-grained environmental change, Online learning (control)
#B: Population, Fine-grained environmental change, Batch learning (default)
#C: Population, Coarse-grained environmental change, Online learning (low mut, fast change)
#D: Hill Climbing, Coarse-grained environmental change, Online learning (control)
#E: Hill Climbing, Fine-grained environmental change, Batch learning (default)
#F: Hill Climbing, Coarse-grained environmental change, Online learning (low mut, fast change)
#G: Hill Climbing, Overfitting, Complex Model
#T: Testing, no predefined parameters
EXPERIMENT_CODE = 'D'