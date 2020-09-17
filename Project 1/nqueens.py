# pip install numpy
import numpy as np
# pip install tqdm
from tqdm import tqdm

class GeneticAlgorithm():

    """ Genetic Algorithm to solve the N-Queens Problem. """

    def __init__(self, n, m, max_gen = 1000, mut_prob = 0.8, cross_prob = 0.8, k = 10):

        """
            n: Number of queens on the chessboard
            m: Number of individuals in the population
            n_gen: Number of generations
            max_gen: Maximum number of generations
            mut_prob: Mutation probability
            cross_prob: Crossover probability
            k: Sample size in parent selection
        """

        self.n = n
        self.m = m
        self.n_gen = 0
        self.max_gen = max_gen
        self.mut_prob = mut_prob
        self.cross_prob = cross_prob
        self.k = k
        self.best_solution = np.zeros((self.n,))
        self.best_fitness = float('Inf')

    def initial_population(self):
    
        """ Random population initialization. """

        # Empty population
        population = list()
        fitness = np.zeros((self.m,))
        
        # Generating initial population
        for i in range(self.m):
            solution = list(np.random.choice(range(1, self.n+1), size = self.n, replace = False))
            fitness[i] = self.fitness_function(solution)
            population.append(solution)
        
        population = np.reshape(population, newshape = (self.m, self.n))
        
        return population, fitness

    def fitness_function(self, solution):
        
        """ Returns the amount of collisions for a given solution (permutation).
            The maximum amount of collisions that can occur is n(n-1)/2.
            For a 4x4 chessboard, the maximum amount of collisions is 6, corresponding 
            to the situation in which all queens are in the same diagonal. """

        # Fitness
        fitness = 0
        
        # 0 to n-1
        for i in range(self.n):
            for j in range(self.n):
                if abs(i-j) == abs(solution[i] - solution[j]) and i != j:
                    fitness = fitness + 1
        fitness = fitness/2
        
        return fitness

    def cut_and_crossfill_crossover(self, parent_a, parent_b):
        
        """ One crossover point is selected, till this point the permutation is copied from the first parent, 
            then the second parent is scanned and if the number is not yet in the offspring it is added.
            The reverse also happens to generate two children. """
        
        assert (len(parent_a) == len(parent_b)), "Parents with different lengths"

        # Offspring
        offspring_a = parent_a.copy()
        offspring_b = parent_b.copy()

        prob = float(np.random.uniform(low = 0, high = 1))

        if prob < self.cross_prob:
      
            # Selecting crossover point
            point = int(np.random.randint(low = 1, high = self.n-1, size = 1))
            
            # First offspring
            head = parent_a[:point]
            tail = [i for i in parent_b if i not in head]
            offspring_a = np.concatenate([head, tail], axis = 0)
            
            # Second offspring
            head = parent_b[:point]
            tail = [i for i in parent_a if i not in head]
            offspring_b = np.concatenate([head, tail], axis = 0)
        
        return offspring_a, offspring_b    

    def mutation(self, parent):
        
        """ Two elements will be exchanged with a probability of mut_prob. """
        
        # Offspring
        offspring = parent.copy()
        
        prob = float(np.random.uniform(low = 0, high = 1))
        
        if prob < self.mut_prob:
                    
            # Selecting mutation points
            points = np.random.choice(range(self.n), size = 2, replace = False)
            
            # Swap
            offspring[points[0]], offspring[points[1]] = offspring[points[1]], offspring[points[0]]
            
        return offspring        


    def parent_selection(self, population, fitness):
        
        """ Selects two parents from a sample of the population. """

        # Candidate solutions
        idxs = np.random.choice(range(self.m), size = self.k, replace = False)
        candidate_solutions = population[idxs]
        candidate_fitness = fitness[idxs]        
        
        # Selects the two best individuals for crossing
        best_idxs = np.argsort(candidate_fitness)[:2]
        parent_a = candidate_solutions[best_idxs[0]]
        parent_b = candidate_solutions[best_idxs[1]]
        
        return parent_a, parent_b

    def survivor_selection(self, population, fitness):

        """ Selection of individuals who will remain in the population. """
        
        # Selects the two worst individuals in the population
        idxs = np.argsort(fitness)[-2:]
        
        # Eliminates the two worst individuals in the population
        population = np.delete(population, idxs, axis = 0)
        fitness = np.delete(fitness, idxs)
        
        return population, fitness

    def solve(self):

        """ Solves the N-Queens Problem. """
        
        # Initializes population with random candidate solutions
        population, fitness = self.initial_population()
        
        # Progress bar
        pbar = tqdm(total = self.max_gen, desc = 'Generations')
               
        # Until the maximum number of generations is reached or the optimal solution is found
        while self.best_fitness > 0 and self.n_gen < self.max_gen:
            
            # Selects parents
            parent_a, parent_b = self.parent_selection(population, fitness)

            # Recombines pairs of parents
            offspring_a, offspring_b = self.cut_and_crossfill_crossover(parent_a, parent_b)

            # Mutates the resulting offspring
            offspring_a = self.mutation(offspring_a)
            offspring_b = self.mutation(offspring_b)

            # Evaluates new candidates
            fitness_a = self.fitness_function(offspring_a)
            fitness_b = self.fitness_function(offspring_b)

            # Adds new individuals to the population
            population = np.vstack([population, [offspring_a, offspring_b]])
            fitness = np.append(fitness, [fitness_a, fitness_b])
            
            # Selects individuals for the next generation
            population, fitness = self.survivor_selection(population, fitness)

            # Updates new best solution
            self.best_fitness = np.min(fitness)
            self.best_solution = population[np.argmin(fitness)]
            
            # Increases number of generations
            self.n_gen = self.n_gen + 1
            pbar.update(1)
            
        pbar.close()

if __name__ == '__main__':

    GA = GeneticAlgorithm(n = 8, m = 100, max_gen = 500, mut_prob = 0.8, cross_prob = 0.8, k = 10)
    GA.solve()
    print('Best solution:', GA.best_solution)
    print('Best fitness:', GA.best_fitness)
