# #libraries

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from IPython.display import display, SVG
import re
from difflib import SequenceMatcher
import random
import itertools
import time
import Levenshtein as lev

#import torch
#torch.device('cuda')

def generate_random_sequence(sequence_length):
    
    """
    Objective: 
    --------------------------------------
    Generate a random peptide sequence of desired length.
    
    Input
    --------------------------------------
    Desired length of random sequence (int)
    
    Output 
    --------------------------------------
    Peptide sequence ---------------> (str)
    """
    
    amino_acid_list = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    
    return ''.join(random.choices(amino_acid_list, k = sequence_length))

def mutate_seq(sequence, n_mutations = 1, type = 'uniform'): 
    
    """
    Objective: 
    --------------------------------------
    Make a random single mutation in a given sequence. The location as well as the mutation is sampled from
    a uniform random distribution.
    
    Input 
    --------------------------------------
    sequence (str)
    
    Output
    --------------------------------------
    sequence (str) 
    """

    if type == 'uniform' and n_mutations > 1:

        n_mutations = random.randint(0, n_mutations)
    
    amino_acid_list = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    
    sequence_list = list(sequence)

    for i in range(n_mutations):
    	index = random.randint(0, len(sequence) - 1)
    	sequence_list[index] = random.choice(amino_acid_list)
 
    return ''.join(sequence_list)

def evaluate_move(old_fitness, new_fitness, 
                  temp = 0.25):
    
    """
    Objective:
    --------------------------------------
    Evaluate whether a move should be accepted or rejected based on the fitness, using the standard
    MH_MCMC criterion on the bolzmann distribution. IMPORTANT: This assumes we want to maximize, not minimize, 
    our fitness function.
    
    Example: 
    --------------------------------------
    If using toy fitness model 1, our old fitness is 0.5, and new fitness is 0.4, then the move will be accepted 
    with probability 0.67 with our choice of temperature at 0.25. We can (and should) play around with the temperature
    since that would determine how liberal our model would be to accepting new moves. Conventionally, the temperature
    is set so that about 23 percent of unfavorable moves are accepted.
    
    Input
    --------------------------------------
    
    old_fitness --------> (float) represents the fitness of the sequence pre-move
    new_fitness --------> (float) represents the fitness of the suggested sequence (suggested move) 
    
    Output
    --------------------------------------
    
    (bool) Decision to accept or reject a suggested move
    """
    
    if new_fitness > old_fitness:
        
        return True
    
    if new_fitness <= old_fitness: 
        
        #return (np.exp((new_fitness - old_fitness)/temp))
        return (random.random() < (np.exp((new_fitness - old_fitness)/temp)))
    
def evaluate_swap(fitness_1, fitness_2, temp_1 = 0.25, temp_2 = 0.4):
    
    """
    Objective:
    --------------------------------------
    Evaluate whether a swap between two chains at different temperatures should be accepted or rejected based on the fitness 
    assuming the fitness is temperature independant (which is true for us). Note that temp_2 > temp_1 (for the equation to hold true).
    Once again, this is assuming we want to maximize, NOT minimize, our fitness function. 
    
    In the identical temperature case, a swap makes no sense as the evolution of all chains would be evaluated using the same criterion. 
    
    Input 
    --------------------------------------
    fitness_1 -------> (float) (fitness of sequence in the first chain)
    fitness_2 -------> (float) (fitness of sequence in the second chain)
    temp_1    -------> (float) (temperature of first (lower temp) chain)
    temp_2    -------> (float) (temperature of second (higher temp) chain)

    Output 
    --------------------------------------
    (bool) (decision to accept or reject a move)
    """
    
    swap_probability = min(1, (np.exp(-(fitness_1 - fitness_2)*(1/temp_1 - 1/temp_2))))
    
    #print(swap_probability)
    
    if swap_probability == 1: 
        
        return True
    
    else: 
        
        return (random.random() < swap_probability)


def toy_fitness_1(sequence, noise = 0.05):
    
    """
    Objective: 
    ---------------------------------
    Output a number that represents a hypothetical fitness value that favors the presence of alanines and glycines in a given sequence. 
    Maximum fitness is achieved when the entire sequence consists entirely of alanines and glycines. 
    In this case, the fitness of our sequence would be equal to 1. 
    
    Input:
    ---------------------------------
    sequence --------> (str)
    
    
    Output:
    ---------------------------------
    fitness ---------> (float) 
    
    Additional information:
    ---------------------------------
    Max Fitness: 1
    Min Fitness: (-0.05, 0) (with noise sampled from a uniform random distribution)
    """
    
    length = len(sequence) 
    alanine_freq = len(re.findall('A', sequence))
    glycine_freq = len(re.findall('G', sequence))
    total_ala_gly = alanine_freq + glycine_freq
    total_non_des = length - total_ala_gly       #total non_desired residues
    fitness = (total_ala_gly - (noise*total_non_des*random.random()))/length
    
    return fitness

def toy_fitness_2(sequence, noise = 0.2):
    
    """
    Objective: 
    ---------------------------
    Output a number that represents a hypothetical fitness value that favors the presence of 'AG' and 'VC' in a given sequence. 
    Maximum fitness is achieved when the entire sequence consists entirely of 'AG' or 'VC' stacked after one another. In this case, 
    we have scaled our maximum fitness to 1 for simplicity. 
    
    Input: 
    ---------------------------
    sequence (str)
    
    Output: 
    ---------------------------
    fitness (float) 
    
    Additional Info:
    Max Fitness: 1
    Min Fitness: (-0.2, 0) (with noise sampled from a uniform random distribution)
    """
    
    length = len(sequence) 
    ag_freq = len(re.findall('AG', sequence))
    vc_freq = len(re.findall('VC', sequence))
    total_ag_vc = ag_freq + vc_freq
    total_non_des = length - total_ag_vc       #total non_desired residues
    fitness = (4*total_ag_vc**2 - (noise*total_non_des*random.random()))/length**2
    
    return fitness

def MH_MCMC(sequence, fitness_model, no_of_iterations, 
            n_mutations = 4,
            temp = 0.025, 
            convergence_known = False, 
            threshold = 0.95, 
            plot_fitness = False, 
            plot_sensitivity = 50,
            verbose = False):
    
    """
    Objective:
    --------------------------------
    Implement the MH_MCMC algorithm for a single sequence at a given temperature by evaluating mutations at every
    iteration. 
    
    In practice, the fitness function maximum is usually 
    unknown, but we include functionality to stop the algorithm if we reach a convergence threshold for a known fitness functions (like
    an artificial, toy fitness function, or an output of a XGBoost or CNN model with a maximum fitness of 1). 
    
    Input 
    --------------------------------
    sequence (str) ---------------> (initial sequence to start with)
    fitness_model (function) -----> (a function that takes a sequence as input, and returns a float that represents some kind of fitness)
    no_of_iterations (int) -------> (the maximum length of our MH_MCMC algorithm)
    n_mutations (int) ------------> (the maximum number of allowed mutations in the sequence (see mutate_seq documentation for more info))
    temp (float) -----------------> (decides how liberally the algorithm accepts moves that decrease the fitness)
    convergence_known (bool) -----> (if True, this assumes that the maximum of the fitness function is known)
    threshold (float) ------------> (if the convergence_known flag is true, the algorithm will stop if the fitness 
                                     function exceeds this value)
    plot_fitness (bool) ----------> (if True, the evolution of the fitness will be plotted)
    plot_sensitivity (int) -------> (frequency of recording the fitness history is determined by this parameter)
    
    Output 
    --------------------------------
    (sequence (str), fitness (float), fitness_history (list)) 
    """
    fitness_history = list()
    start_time = time.time()
    old_fitness = None

    for i in range(no_of_iterations):
        
        if old_fitness == None:
            old_fitness  = fitness_model(sequence)
            
        new_sequence = mutate_seq(sequence, n_mutations = n_mutations)
        new_fitness  = fitness_model(new_sequence)
        
        if evaluate_move(old_fitness, new_fitness, temp):
        
            sequence = new_sequence
            old_fitness = new_fitness
        
        #for toy functions and fitness functions with known/achievable maximums
        if (new_fitness > threshold and convergence_known is True):
            
            if verbose is True:
                
                print('Convergence achieved at i =', i, 'with fitness', new_fitness)
            
            break
        
        #record fitness evolution if requested with plot_fitness flag
        if plot_fitness is True:
            
            if i % plot_sensitivity == 0: 
                
                fitness_history.append(old_fitness)
                
        if verbose is True and (i%100 == 0): 
            
            print(f'Progress: {round(i*100/no_of_iterations, 2)}%, Iteration Rate: {round((i/(time.time()-start_time)),2)}/s \n')
            print(f'Sequence:{sequence}, Stored Fitness {old_fitness:.3g}, Current Fitness: {new_fitness:.3g} \n', end = '\r')
    
    if plot_fitness is True:
        
        plt.figure(figsize = (16, 12))
        
        plt.plot(np.arange(len(fitness_history)), fitness_history)
        plt.xlabel(f'Iterations (x{plot_sensitivity})')
        plt.ylabel('Fitness')
        plt.title(f'MH_MCMC Fitness Evolution with T = {temp}')
        plt.savefig('my_plot.png')
        
    return (sequence, fitness_model(sequence), fitness_history)

def crossover_sequences(sequence_1, 
                        sequence_2):
    
    """
    Objective: 
    ---------------------------
    Generate two new strings by crossing over two given strings. Note, the crossover point is 
    generated using a uniform distribution.   
    
    Input 
    ---------------------------
    sequence_1 (str)
    sequence_2 (str) 
    
    Output 
    --------------------------- 
    new_sequence_1 (str), new_sequence_2 (str)
    """
    
    #sequence_1 = sequence_list[index_of_first_sequence]
    #sequence_2 = sequence_list[index_of_second_sequence]
    
    break_point = random.randint(0, len(sequence_1))
    new_sequence_1 = sequence_1[:break_point] + sequence_2[break_point:]
    new_sequence_2 = sequence_2[:break_point] + sequence_1[break_point:]

    return new_sequence_1, new_sequence_2

def test_evaluate_crossover(old_fitness_pair, new_fitness_pair,
                            temperature_pair):
    
    """
    Objective:  
    ---------------------------
    Writing this to test the results of the crossover evaluation for a given temperature, old, and new fitness pairs. 
    
    Input: 
    ---------------------------
     
    old_fitness_pair ------------> (tuple) of (floats) with fitness of sequences prior to being crossedover. 
    new_fitness_pair ------------> (tuple) of (floats) with fitness of crossed-over sequences post crossover.
    temperature_pair ------------> (tuple) of (floats) the temperatures of the chains which were crossedover.
    
    Output: 
    swap_probability ------------> (float) probability of move acceptance
    """
    
    old_fitness_1 = old_fitness_pair[0]
    old_fitness_2 = old_fitness_pair[1]
    
    if old_fitness_1 < old_fitness_2: 
        
        temp_var = old_fitness_1
        old_fitness_1 = old_fitness_2 
        old_fitness_2 = temp_var
    
    new_fitness_1 = new_fitness_pair[0]
    new_fitness_2 = new_fitness_pair[1]
    
    if new_fitness_1 < new_fitness_2: 
        
        temp_var = old_fitness_1
        new_fitness_1 = new_fitness_2 
        new_fitness_2 = temp_var 
    
    temp1 = temperature_pair[0]
    temp2 = temperature_pair[1]
    
    swap_probability = min(1, np.exp((new_fitness_1 - old_fitness_1)/temp1 - (new_fitness_2 - old_fitness_2)/temp2))

    return (swap_probability)

def perform_and_evaluate_swap(sequence_list, temperature_list, fitness_list,
                              fitness_model, swap_pair):
    
    """
    Objective:
    ---------------------------
    Perform a swap move for the given swap pair, and then accept or reject the move. The chain swap is
    evaluated according to a replica exchange/parallel tempering algorithm. 
    
    Input:
    ---------------------------
    
    sequence_list -----------------> (list) of (str) before the swap move
    temperature_list --------------> (list) of (float)
    fitness_model -----------------> (function) that returns a (float) between 0 and 1 representing the fitness
    swap_pair ---------------------> (tuple) containing indices of sequences to be crossedover
    
    Output: 
    ---------------------------
    sequence_list -----------------> (list) of (str) if the swap move is accepted, then this list is 
    updated to include the swapped sequences, if not, then the sequence_list is unchanged
                                     
    """
    
    sequence_1 = sequence_list[swap_pair[0]]
    sequence_2 = sequence_list[swap_pair[1]]
    
    fitness_1 = fitness_model(sequence_1)
    fitness_2 = fitness_model(sequence_2)
    
    if evaluate_swap(fitness_1, fitness_2, 
                     temperature_list[swap_pair[0]], 
                     temperature_list[swap_pair[1]]):
        
        sequence_list[swap_pair[0]] = sequence_2
        sequence_list[swap_pair[1]] = sequence_1
        
        fitness_list[swap_pair[0]] = fitness_2
        fitness_list[swap_pair[1]] = fitness_1
        
        return sequence_list, fitness_list
    
    else:
        
        return sequence_list, fitness_list
        

def perform_and_evaluate_crossover(sequence_list,
                                   temperature_list, fitness_model, fitness_list,
                                   crossover_pair):
    
    """
    Objective:
    ---------------------------
    Perform a crossover move for the given crossover pair, and then accept or reject the move. Sequences are crossed 
    over as in the function  "crossover_sequences", and the crossover is evaluated according to an 
    adapted MCMC criterion developed by Liang and Wong. 
    
    Input:
    ---------------------------
    
    sequence_list -----------------> (list) of (str) before the crossover move
    temperature_list --------------> (list) of (float)
    fitness_model -----------------> (function) that returns a (float) between 0 and 1 representing the fitness
    crossover_pair ----------------> (tuple) containing indices of sequences to be crossedover
    
    Output: 
    ---------------------------
    sequence_list -----------------> (list) of (str) if the crossover move is accepted, then this list is updated to include 
    the new crossedover sequences, if not, then the sequence_list is unchanged                                     
    """
    
    sequence_1 = sequence_list[crossover_pair[0]]
    sequence_2 = sequence_list[crossover_pair[1]]
    
    fitness_1 = fitness_list[crossover_pair[0]]
    fitness_2 = fitness_list[crossover_pair[1]]
    
    if fitness_1 < fitness_2: 
        
        temp_var = fitness_1
        fitness_1 = fitness_2 
        fitness_2 = temp_var
        
        temp_seq = sequence_1
        sequence_1 = sequence_2 
        sequence_2 = temp_seq
        
    new_sequence_1, new_sequence_2 = crossover_sequences(sequence_1, sequence_2)
    
    new_fitness_1 = fitness_model(new_sequence_1)
    new_fitness_2 = fitness_model(new_sequence_2)
    
    if new_fitness_1 < new_fitness_2: 
        
        temp_var = new_fitness_1
        new_fitness_1 = new_fitness_2 
        new_fitness_2 = temp_var
        
        temp_seq = new_sequence_1
        new_sequence_1 = new_sequence_2 
        new_sequence_2 = temp_seq
    
    temp1 = temperature_list[crossover_pair[0]]
    temp2 = temperature_list[crossover_pair[1]]
    
    swap_probability = min(1, np.exp((new_fitness_1 - fitness_1)/temp1 - (new_fitness_2 - fitness_2)/temp2))
    
    if swap_probability == 1: 
        
        sequence_list[crossover_pair[0]] = new_sequence_1
        sequence_list[crossover_pair[1]] = new_sequence_2
        
        fitness_list[crossover_pair[0]] = new_fitness_1
        fitness_list[crossover_pair[1]] = new_fitness_2
        
        return sequence_list, fitness_list
    
    else:
    
        if (random.random() < swap_probability):
            
            sequence_list[crossover_pair[0]] = new_sequence_1
            sequence_list[crossover_pair[1]] = new_sequence_2
            
            fitness_list[crossover_pair[0]] = new_fitness_1
            fitness_list[crossover_pair[1]] = new_fitness_2
        
            return sequence_list, fitness_list
        
        return sequence_list, fitness_list

def generate_fitness_list(sequence_list, fitness_model):
    
    """
    Objective: 
    Given a list of sequences, calculate the fitness of each of the sequences using the provided fitness function. 
    
    Input:
    sequence_list ((list) of (str))
    fitness_model (function with *arg = sequence (str))
    
    Output:
    fitness_list ((list) of (float))
    """
    
    fitness_list = list()
    
    for sequence in sequence_list:
        
        fitness_list.append(fitness_model(sequence))
        
    return fitness_list

def generate_pairs_consecutive(no_of_events, length_of_sequence_list):
    
    """
    Objective: 
    Generate a tuple or list of tuples containing unique pairs for each swap event. The requirement in a swap event is that the indices must be
    consecutive. 
    
    Example: 
    
    A list of 4 sequences has 3 possible pairs (1,2),(2,3),(1,3); if the no_of_events is 2, this function will sample 2 of these 3 pairs
    using a uniform random distribution
    
    Input ----- no_of_events (int) (number of crossover pairs to generate)
                length_of_sequence_list (int)
                
    Output ---- [(chain_1_index,chain_2_index), (.,.), ...] (list) of (tuples)
    """
    
    pair_list = list()
    
    for i in range(length_of_sequence_list - 1):
        
        pair_list.append((i, i+1))
    
    if no_of_events >= len(pair_list):
    
        return pair_list
    
    else:
        
        pair_choices = random.sample(pair_list, k = no_of_events)
    
        return pair_choices
    
    return None

def generate_pairs_random(length_of_sequence_list, no_of_events, identical_temperatures = False):
    
    """
    Objective: 
    Generate a tuple or list of tuples containing unique pairs for each crossover event. 
    
    Example: 
    
    A list of 4 sequences has 3 possible pairs (1,2),(2,3),(1,3); if the no_of_events is 2, this function will sample 2 of these 3 pairs
    using a uniform random distribution
    
    Input ----- no_of_events (int) (number of crossover pairs to generate)
                length_of_sequence_list (int)
                
    Output ---- [(chain_1_index,chain_2_index), (.,.), ...] (list) of (tuples)
    """
    
    pair_list = [i for i in itertools.combinations(np.arange(0, length_of_sequence_list), 2)]
  
    #print(pair_list)
    
    pair_choices = random.sample(pair_list, k = no_of_events)
    
    return pair_choices

def emc_v2(sequence_list, fitness_model, no_of_iterations, 
           temperature_list,
           crossover_rate = 0.5,
           mhmcmc_repeats = 1,
           n_mutations = 4,
           crossover_events = 1,
           swap_events = 1, 
           plot_fitness = False,
           convergence_known = False, 
           delay_convergence = False,
           delay_by = 100,
           threshold = 0.95, 
           plot_sensitivity = 100,
           save_png = False,
           png_title = f'my_plot.png'):

    """
    Objective: 
    ---------------------------------------------
    Implement the EMC algorithm using N sequences at a given temperature ladder. This function assumes that the 
    maximum fitness is known (since we are using toy fitness functions for now), in practice, this is usually unknown. This function
    also assumes that the algorithm wants to maximize, not minimize, the fitness function. 

    Input 
    -------------------------------------------- 
    sequence_list ((list) of (str)) (initial list of sequences to start with)
    fitness_model (function) (a function that takes a sequence as input, and returns a float that represents some kind of fitness)
    no_of_iterations (int) (the maximum length of our EMC algorithm)
    temperature_list ((list) of (float)) (lists the temperatures at which each MH_MCMC chain is run at; they can be identical
                                          simplest case, or ladder like for RE/PT or EMC)
                                          ***must be the same length as sequence_list! 
    mhmcmc_repeats (int) the number of times an MHMCMC step is repeated before the end of the primary iteration, default is 1
    n_mutations (int) the maximum number of mutations to make in an MHMCMC step (see mutate_seq for more info)
    crossover_rate (float) (proportion of iterations that result in a crossover event being evaluated, should be between 0 (for no crossover) and 0.5)
    crossover_events (int) (number of crossover or swap operations to consider per primary iteration, default is one)
    swap_events (int) (number of crossover or swap operations to consider per primary iteration, default is one)
    plot_fitness (bool) (if True, the evolution of the fitness will be plotted)
    convengence_known (bool) (if True, the algorithm stops when the lowest temperature chain's fitness is above the 'threshold' variable, see also delay_convergence)
    delay_convergence (bool) (if True, the algorithm will pretend it does not know the convergence threshold until the number of iterations exceeds the
                              'delay_by' variable, this only works if convergence_known is False)
    delay_by (int) (see delay_convergence)
    threshold (float) (see convergence_known)
    plot_sensitivity (int) (frequency of recording the fitness history is determined by this parameter)
    save_png (bool) flag to indicate whether or not we should save our png (from plot_fitness) to the parent directory 
    png_title (str) png name to save with if save_png and plot_fitness are both true
    
    Output 
    ---------------------------------------------
    (updated_sequence_list ((list) of (str)), 
    fitness_history_complete ((list) of (list) of (float)),  
    """
    
    current_fitness = generate_fitness_list(sequence_list, fitness_model)
    fitness_history_complete = list()
    
    start_time = time.time()
    
    for i in range(no_of_iterations + 1):
    
        #decide whether to mutate or to crossover
        
        if random.random() > crossover_rate:
            
            for j in range(len(sequence_list)):
                sequence_list[j], current_fitness[j] = MH_MCMC(sequence_list[j], fitness_model, mhmcmc_repeats, 
                                                                    n_mutations = n_mutations, temp = temperature_list[j])[:2]
        
        else:
            
            pair_list = generate_pairs_random(len(sequence_list), crossover_events)
            
            for pair in pair_list: 
                
                sequence_list, current_fitness = perform_and_evaluate_crossover(sequence_list, temperature_list, fitness_model, current_fitness, pair)
        
        #evaluate swaps  
        
        swap_pair_list = generate_pairs_consecutive(len(sequence_list), swap_events)
        
        for pair in swap_pair_list:
            
            sequence_list, current_fitness = perform_and_evaluate_swap(sequence_list, temperature_list, current_fitness, fitness_model, pair)
            
        #update fitness history list
        if (i % plot_sensitivity) == 0: 
            
            fitness_history_complete.extend(current_fitness)

            print(f'Progress: {(i*100/no_of_iterations):.3g}%, Fitness: {(max(current_fitness)):.3g}, Iteration_rate: {(i/(time.time()- start_time)):.3g}', end = '\r')
            
        #set convergence_known to True after set iterations if delay_convergence is True
        if delay_convergence is True and i > delay_by: 
            
            convergence_known = True

        
        if convergence_known is True: 
            
            if max(current_fitness) > threshold: 
                
                print(f'Convergence achieved at i = {i} \n \n')
                
                break
    
    #recast as np array for recording/plotting
    fitness_history_complete = np.asarray(fitness_history_complete)
    shape_dim_1 = int(len(fitness_history_complete)/len(temperature_list))
    shape_dim_2 = int(len(temperature_list))
    fitness_history_complete = np.reshape(fitness_history_complete, (shape_dim_1, shape_dim_2))
    
    if plot_fitness is True:
        
        #print(len(fitness_history_complete)/len(temperature_list), len(temperature_list))
        #print(fitness_history_complete.shape)
                                              
        plt.figure(figsize = (16, 12))
        
        for k in range(len(fitness_history_complete[0][-3:])):
        
            plt.plot(np.arange(len(fitness_history_complete)), fitness_history_complete[:,k], label = f'Sequence_T: {temperature_list[k]}')
            
            if save_png is True:
                plt.savefig(png_title)
        
        #plt.plot(np.arange(len(fitness_history_list[0])), fitness_history_list[1], label = 'Sequence_2')
        plt.legend()
        plt.xlabel(f'Iterations (x{plot_sensitivity})')
        plt.ylabel('Fitness')
        plt.title(f'EMC/PT Fitness Evolution of lowest 1-3 chains')
        
    return(sequence_list, 
           fitness_history_complete) 

def entropy_difference_HMCMC(initial_sequence, no_of_iterations, mutation_rate = 5):
    
    """
    Objective: 
    ---------------------------------
    Simulate a metropolis hastings monte carlo simulation for the given parameters and record the entropy change for each 
    sequence in the list for each iteration. Return this 'entropy change history' as a list.
    
    Input: 
    ---------------------------------
    initial_sequence (str) 
    no_of_iterations (int) 
    mutation_rate (int) the maximum number of point mutations allowed in the MHMCMC step
    
    Output: 
    ---------------------------------
    (list) of the shannon entropy changes from iteration to iteration of the HMCMC run.
    """


    entropy_list = list()

    for i in range(no_of_iterations): 
    
        entropy_list.append(calculate_entropy(initial_sequence))
        initial_sequence = mutate_seq(initial_sequence, mutation_rate)

    entropy_list_2 = np.asarray(cyclic_perm(entropy_list))
    entropy_list   = np.asarray(entropy_list)

    entropy_diff_history  = np.abs(entropy_list - entropy_list_2)

    return entropy_diff_history 

def entropy_difference_crossover_EMC(initial_sequence_list, no_of_iterations, crossover_rate = 0.5, crossover_events = 1, mutation_rate = 5):
    
    """
    Objective: 
    ---------------------------------
    Simulate an evolutionary monte carlo simulation for the given parameters and record the entropy change for each sequence in the list for each iteration.
    Then, calculate the maximum entropy change per iteration and return that as a list which represents the magnitude of the entropy change over the 
    course of the simulation. This simulation assumes that each iteration is accepted in the run, which is fine because we want to calculate the 
    magnitude of the entropy changes from iteration to iteration. 
    
    Input: 
    ---------------------------------
    initial_sequence_list (list) of (str) containing the starting sequences for the EMC run. 
    no_of_iterations (int) 
    crossover_rate (float) between 0 and 1, default set to 0.5. This represents the ratio of iterations (roughly that will enter the crossover process) 
    crossover_events (int) default set to 1. The number of pairs for which a crossover will be performed. 
    mutation_rate (int) the maximum number of point mutations allowed in the MHMCMC step of the EMC run 
    
    Output: 
    ---------------------------------
    (list) of the maximum shannon entropy changes from iteration to iteration of the EMC run.
    """

    length_of_sequence_list = len(initial_sequence_list)
    
    for iteration in range(no_of_iterations):
    
        pair_list = generate_pairs_random(len(initial_sequence_list), crossover_events)
    
        entropy_before_list = np.zeros(length_of_sequence_list)
        entropy_after_list = np.zeros(length_of_sequence_list)
        
        for i in range(length_of_sequence_list):
                entropy_before_list[i] = calculate_entropy(initial_sequence_list[i])
    
        if random.random() > crossover_rate:
            
            for i in range(len(initial_sequence_list)):
        
                initial_sequence_list[i] = mutate_seq(initial_sequence_list[i], mutation_rate)
        
        else:
        
            for pair in pair_list:
    
                l = pair[0]
                m = pair[1]
    
                initial_sequence_list[l], initial_sequence_list[m] = crossover_sequences(initial_sequence_list[l], initial_sequence_list[m])
        
        #print(initial_sequence_list)
        
        for i in range(length_of_sequence_list):
                entropy_after_list[i] = calculate_entropy(initial_sequence_list[i])
    
        entropy_diff_list = np.abs(entropy_after_list - entropy_before_list)
        entropy_diff_history.append(max(entropy_diff_list))
    
    return entropy_diff_history

#def emc_v1(sequence_1, sequence_2, fitness_model, no_of_iterations_primary, 
#            no_of_iterations_secondary, temp = 0.025, plot_fitness = False, 
#            convergence_fitness = 0.95, plot_sensitivity = 50):

#     """
#     Objective: 
#     Implement the EMC algorithm using two sequences at a given temperature. This function assumes that the 
#     maximum fitness is known (since we are using toy fitness functions for now), in practice, this is usually unknown. 

#     This is the simplest implementation, i.e. everything happens at the same temperature.

#     Input 
#     ---------------------------------- 
    
#     sequence_1 (str) (initial sequence to start with)
#     sequence_2 (str)
#                 fitness_model (function) (a function that takes a sequence as input, and returns a float that represents some kind of fitness)
#                 no_of_iterations_primary (int) (the maximum length of our EMC algorithm)
#                 no_of_iterations_primary (int) (the maximum length of our MH_MCMC algorithm)
#                 plot_fitness (bool) (if True, the evolution of the fitness will be plotted)
#                 plot_sensitivity (int) (frequency of recording the fitness history is determined by this parameter)
    
#     Output ---- (sequence_1 (str), sequence_2 (str), 
#                  fitness_1 (float), fitness_2 (float), 
#                  fitness_history_1 (list), fitness_history_2 (list))
#     """
    
#     fitness_history_1 = list()
#     fitness_history_2 = list()
    
#     fitness_1 = 0
#     fitness_2 = 0
    
#     for i in range(no_of_iterations_primary):
    
#         #implement independant MH_MCMC
#         sequence_1 = MH_MCMC(sequence_1, fitness_model, no_of_iterations_secondary, temp = temp)[0]
#         sequence_2 = MH_MCMC(sequence_2, fitness_model, no_of_iterations_secondary, temp = temp)[0]
        
#         #generate two new sequences by crossing over 
#         break_point = random.randint(0, len(sequence_1))
#         new_sequence_1 = sequence_1[:break_point] + sequence_2[break_point:]
#         new_sequence_2 = sequence_2[:break_point] + sequence_1[break_point:]
        
#         old_fitness_1 = fitness_model(sequence_1)
#         old_fitness_2 = fitness_model(sequence_2)
        
#         new_fitness_1  = fitness_model(new_sequence_1)
#         new_fitness_2  = fitness_model(new_sequence_2)
        
#         if (old_fitness_1 > convergence_fitness and
#             old_fitness_2 > convergence_fitness):
            
#             print('Convergence achieved at i =', i, 'with fitness', old_fitness_1, 'and', old_fitness_2)
#             fitness_1 = old_fitness_1
#             fitness_2 = old_fitness_2
#             break
    
#         if (evaluate_move(old_fitness_1, new_fitness_1, temp) and
#            evaluate_move(old_fitness_2, new_fitness_2, temp)):
        
#             sequence_1 = new_sequence_1
#             sequence_2 = new_sequence_2
        
#             #convergence
#             if (new_fitness_1 > convergence_fitness and
#                new_fitness_2 > convergence_fitness):
            
#                 print('Convergence achieved at i =', i, 'with fitness', new_fitness_1, 'and', new_fitness_2)
#                 fitness_1 = new_fitness_1
#                 fitness_2 = new_fitness_2
                
#                 break
        
#         if plot_fitness is True:
            
#             if i % plot_sensitivity == 0: 
                
#                 fitness_history_1.append(old_fitness_1)
#                 fitness_history_2.append(old_fitness_2)
    
#     if plot_fitness is True:
        
#         plt.plot(np.arange(len(fitness_history_1)), fitness_history_1, label = 'Sequence_1')
#         plt.plot(np.arange(len(fitness_history_2)), fitness_history_2, label = 'Sequence_2')
#         plt.legend()
#         plt.xlabel(f'Iterations (x{plot_sensitivity})')
#         plt.ylabel('Fitness')
#         plt.title(f'MH_MCMC Fitness Evolution with T = {temp}')
        
    
#     if (fitness_1 == 0 or fitness_2 == 0): 
        
#         fitness_1 = old_fitness_1
#         fitness_2 = old_fitness_2
    
#     return(sequence_1, sequence_2, 
#            fitness_1, fitness_2, 
#            fitness_history_1, fitness_history_2)
