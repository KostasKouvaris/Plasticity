# -*- coding: utf-8 -*-
################################################################################
######################### NAMING CONVENTIONS ###################################
################################################################################

# Constants are in capital letters separated by underscores
# Classes are annotated using CamelCase notation
# Functions are annotated in lowercase separted by underscores
# Variables are annotated in pascalNotation

################################################################################
############################ IMPORTING #########################################
################################################################################

import os
#import cProfile as profile
from numpy.random import seed
from experiments import run_instance


################################################################################
################################ MAIN ##########################################
################################################################################

def main(): 
    seed(7) #7
    path = '/Plasticity/v5/Experiment_POP_Online_Zero_Cost'
    run_instance(path)

#clear screen
os.system('cls')

#run    
if __name__ == '__main__':
    main()
    #profile.run('main()')


        

