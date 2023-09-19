import numpy as np

def tournament(pop, k=1, lambda_=2, fitness_func=lambda x: np.sum(x)):
    # Select k random indexes from the population
    k_indexes = np.random.randint(0, pop.shape[0], k)
    list_of_individuals = [pop[index] for index in k_indexes]
    
    # Compute the fitness of the selected individuals
    fitness_of_individuals = [fitness_func(individual) for individual in list_of_individuals]
    
    # Sort the individuals based on their fitness
    sorted_indices = np.argsort(fitness_of_individuals)[::-1]

    print(sorted_indices)
    # Get the lambda best individuals
    best_individuals = [list_of_individuals[i] for i in sorted_indices[:lambda_]]
    
    return best_individuals



# Test the function

# fitness_func = lambda x: 1/x

pop = np.random.randint(0, 2, (10, 10))
print(pop)
print()
print(tournament(pop, k=2, lambda_=3))