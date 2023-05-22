# Monte Carlo Fitness Exploration Methods

This repository documents the devlopment and usage of the explore_fitness.py library, intended to provide fitness landscape exploration methods via discrete optimization operations. 

A brief tutorial/documentation on how to use the library is given in the MCMC_Testing_Tutorial.ipynb notebook, and documentation pertaining to all functions in the 'explore_fitness.py' library can be viewed via (help(function_name)). 

We will briefly describe the developed algorithms here, and we will include discussion of potential improvements and refinements wherever possible. 

The operators and the algorithms described in this library are developed to *maximize* a fitness function. 

Before we describe the algorithms, let us first describe the operators we will use in our library:

### Random Mutation(s) Operator

Given a peptide sequence, pick N loci and mutate them randomly. The mutations and their loci are selected using a uniform random distribution. The user can choose the number 'N', and can also choose to mutate anything between 1-N loci at a time (instead of N mutations every iteration). 

### Random Single Point Crossover Operator 

Given two peptide sequences $i$ and $j$ of equal length N, select a number between [2, N-1] from a uniform random distribution which we would use to select a crossover locus. Then, we generate two new sequences, $i_{2}$ and $j_{2}$, such that $i_{2}$ is identical to sequence $i$ prior to our crossover locus, and identical to sequence $j$ post our crossover locus. Similarly, $j_{2}$ is identical to sequence $j$ prior to the crossover locus, and identical to sequence $i$ post the crossover locus. 

In the future, we may choose to add functionality for 2 or 3 point crossovers (multiple crossover loci), as well as crossovers which yield sequences of different lengths. 

### Metropolis Hastings (Boltzmann) Criterion

Let $f_{s}$ represent the fitness of a state/sequence laballed as s. 

$P$(transition to state s) = $min(1, r_{mh})$

where: 

$r_{mh} = exp(\frac {f_{s} - f_{s-1}}{T})$  

### Metropolis Hastings (Boltzmann) Swap Criterion 

$P$(swapping states i and j at $T_{i}$ and $T_{j}$) = $min(1, r_{re})$

where: 

$r_{re} = exp(-(f(i) - f(j))(\frac{1}{T_{i}} - \frac{1}{T_{j}}))$  

### Metropolis Hastings (Boltzmann) Crossover Criterion

Select $i_{1}, i_{2}, j_{1}, j_{2}$ such that $f(i_{1} \geq f(j_{1})$ and $f(i_{2}) \geq f(j_{2})$.

$P$(states $i_{1}$ and $j_{1}$ to $i_{2}$ and $j_{2}$ at $T_{i}$ and $T_{j}$) = $min(1, r_{c})$

where: 

$r_{c} = exp(\frac{(f(i_{2}) - f(i_{1}))}{T_{i}} - \frac{(f(j_{2}) - f(j_{1}))}{T_{j}})$  

Algorithm Details: 

## Metropolis Hasting Markov Chain Monte Carlo (MH_MCMC): 

1. Start with a random sequence $s_{i}$ with fitness $f(s_{i})$. 
2. Propose a new sequence $s_{i + 1}$ with fitness $f(s_{i + 1})$. The new sequence can be proposed via a random single mutation operator, or any other operator of choice. 
3. If $f(s_{i + 1}) > f(s_{i})$, we accept the new sequence and the algorithm starts over. 
4. If $f(s_{i + 1}) > f(s_{i})$, we generate a random number from a uniform random distribution and if that number is smaller than $r_{mh}$ for the proposed move, we accept it and start the algorithm again with our new preposed seqeunce. Else we reject the move and start the algorithm over again with the same seqeunce. 

Steps 3. and 4. are often described as 'the move is accepted with probability $min(1,r)$', where $r$ is the metropolis-hastings criterion, and is usually a number between 0 and 1 for non-optimal moves, the more non-optimal the moves are, the closer that number is to 0. Here, non-optimal simply means a move that decreases the fitness.  

The only hyperparameters in MH_MCMC are temperature and number of iterations. Usually, number of iterations is decided by trying a few numbers and manually tweaking it to find a number where the algorithm is able to converge to a state with an acceptible fitness. This is done by simply plotting the fitness vs. the number of iterations. The temperature (called the temperature because that's what the term represents in the boltzmann distribution of energy states at a finite temperature) is the hyperparameter which decides how liberally non-optimal moves are accepted. A very high temperature will result in a very large proportion of non-optimal moves being accepted (>50-60%). This would make it almost impossible for the algorithm to converge, as the algorithm is unable to reject moves when it is close to an optimum (favoring a large area search). A smaller temperature makes the MH_MCMC algorithm more conservative, thereby only strictly accepting moves that increase the fitness or decrease it but only by a very small amount (favoring a higher resolution search). This makes it easier to move towards an optimum, but restricts the search space to directions where the fitness only increases (somewhat similar to a steepest descent algorithm that is not guided by a derivative). The selection of the temperature is therefore an important task, defining the tradeoff between the resolution (depth) and the width (area) of our search. A rule of thumb is to select a temperature which results in our algorithm reaching convergence but also via accepting about 15-20% of non-optimal moves over the course of the simulation. However, this number is usually problem specific. 

The advantage of MH_MCMC is that it is simple to implement, is not very computationally expensive, and is easy to debug/analyze. The disadvantage is that our search space is highly dependant on our initial starting sequence, and the local landscape around that seqeunce determines whether or not we converge at a fitness that is satisfactory.

## Parallel Tempering/Replica Exchange Monte Carlo (PT/RE MCMC)

0. Define 'N' temperatures in ascending order (usually referred to as a temperature ladder). 
1. Start with 'N' random sequences. 
2. For each 'N' random sequences or 'chains', we propose a new sequence via applying an operator (could be random single mutation, or any other). 
3. For each of those chains, the move to the new proposed sequence in the chain is accepted with probability $min(1,r_{mh})$, i.e. we apply steps 3 and 4 of the MH_MCMC algorithm. The key difference is that each chain is assigned its own temperature which we predefined in step 0.  
4. Two adjacent chains are then selected randomly, and are allowed to 'swap' their sequences with probability $min(1,r_{re})$. 
5. Repeat until convergence and/or total number of iterations. 

PT/RE allows the algorithm the search for an optimum with different thresholds for non-optimal moves. This allows the algorithm to search a larger landscape that what would be possible with a single MH_MCMC chain. The idea is that a higher temperature chain may accept non-optimal moves more liberally, but in the case it moves to a significantly more optimal state, it can then 'swap' that state with a lower temperature one which will then further refine the search near the more optimal sequence. The 'swap' operation always takes place if the lower temperature chain has a less optimal state than the higher temperature chain, and if this condition is not met, then the swap happens with probability proportional to the difference of the fitness measure as well as the difference of the temperatures (as defined in $r_{re}$).

RE/PT is exceptionally useful when we can take advantage of parallel computing to run N chains in parallel. This is a future development objective for this library. This makes RE/PT utilize the MH_MCMC algorithm without having to use a single temperature, thereby qualitatively eliminating the resolution/area tradeoff. However, it is far more computationally expensive that running a single MH_MCMC chain.

The additional hyperparameter in RE/PT is therefore the temperature ladder, which is the temperatures we associate to each chain. He we want the differences in temperature to be such that there is an overlap in the search space between two adjacent temperatures. In other words, we want the temperature differences to be such that the temperature difference dependance in $r_{re}$ yields in a small but non-zero number of non-optimal swaps. In general, if the temperatures are too far apart, the seqeunces almost never swap (they only swap in the rare case when the two chains are selected at the exact moment where the lower temperature chain has is in a less optimal state than the high temperature chain), while if the temperatures are too close, the states are swapped too frequently and the time taken to reach convergence will increase. An optional hyperparameter is the swap_rate, which allows for the evaluation of additional swap pairs per iteration. This can be useful if 'N' is a large number.

## Evolutionary Monte Carlo (EMC)

0. Define 'N' temperatures in ascending order (usually referred to as a temperature ladder). 
1. Start with 'N' random sequences. 
2. We propose a new set of 'N' sequences via either applying a mutation operator with probability (q), or a crossover operator with probability (1-q) to our set of sequences. 
3. If the algorithm chooses to apply the crossover operator, the pair of sequences chosen for a crossover, as well as the number of pairs chosen from crossovers, is predefined. The proposed crossover is then accepted with a probability equal to $min(1, r_{c})$. 
4. Two adjacent chains are then selected randomly, and are allowed to 'swap' their sequences with probability $min(1,r_{re})$. 
5. Repeat until convergence and/or total number of iterations. 

EMC allows significantly large jumps in sequence space due to the crossover operation, which increases our potential search space even more than RE/PT. EMC roughtly mimics genetic information swapping in mating events. 

The additional hyperparameters in EMC are 'q', which determines how often we want to mutate or crossover our N chains, as well as the crossover rate and type, which determines how many chains, and how, we want to crossover. These parameters may be tuned, and are expected to be highly dependant on the geography of the fitness landscape.   

### Future Development Objectives (as of 1/27/2023): 

1. Functions pertaining to temperature determination (static and/or dynamic). 
2. Support for evaluating and generating sequences of different lengths. 
3. Instance dependant multiprocessing for parallel computing. 
