# pip install numpy
import numpy as np
# pip install tqdm
from tqdm import tqdm
# pip install matplotlib
import matplotlib.pyplot as plt

class GeneticAlgorithm():

    """ Genetic Algorithm to find the minimum of Rastrigin's function. """

    def __init__(self, n, m, l, max_eval = 1000, mut_prob = 0.8, cross_prob = 0.8, k = 10):

        """
            n: Number of variables
            m: Number of individuals in the population
            l: Number of bits to encode all variables
            n_eval: Number of function evaluations
            max_eval: Maximum number of function evaluations
            mut_prob: Mutation probability
            cross_prob: Crossover probability
            k: Sample size in tournament selection
        """

        self.n = n
        self.m = m
        self.l = l
        self.n_eval = 0
        self.max_eval = max_eval
        self.mut_prob = mut_prob
        self.cross_prob = cross_prob
        self.k = k
        self.interval = [-5.12, 5.12]
        self.best_solution = np.zeros(((self.n * self.l),))
        self.best_fitness = float('Inf')

    def encoding(self, X):
    
        """ Encoding each real solution to a binary representation. """
        
        B = list()
        
        # For each variable X[i]
        for i in range(self.n):        
            k = int(np.ceil((X[i] - self.interval[0])*(np.power(2,self.l)-1)/(self.interval[1] - self.interval[0])))
            b = bin(k).replace("0b", "")
            b = [float(bi) for bi in b]
            
            if len(b) < self.l:
                b = list(np.zeros((self.l-len(b),))) + b
                
            B.append(b)    
        
        B = np.array(B).reshape((-1))
        
        return B

    def decoding(self, B):
    
        """ Decoding each binary solution to a floating point representation. """
        
        X = np.zeros((self.n,))
        
        # The interval between adjacent values
        delta = (self.interval[1] - self.interval[0])/(np.power(2,self.l)-1)  
        
        # For each variable b in B
        for i in range(self.n):
            b = B[self.l*i:self.l*(i+1)]
            X[i] = self.interval[0] + delta * sum([b[k]*np.power(2,(self.l-k-1)) for k in range(self.l)])

        return X

    def random_initial_population(self):
    
        """ Random population initialization. """
        
        # Empty population
        population = list()
        fitness = np.zeros((self.m,))
            
        # Generating initial population
        for i in range(self.m):
            X = np.random.uniform(low = self.interval[0], high = self.interval[1], size = self.n)
            fitness[i] = self.rastrigin(X)
            B = self.encoding(X)
            population.append(B)
            
        population = np.array(population).reshape((self.m,(self.n*self.l)))
        
        return population, fitness

    def opposition_based_population(self):

        """ Opposition-based population initialization. 
            A novel population initialization method for accelerating evolutionary algorithms 
            (https://www.sciencedirect.com/science/article/pii/S0898122107001344). """

        # Empty population
        population = list()
        fitness = np.zeros((2*self.m,))
            
        # Generating initial population
        i = 0
        while i < 2*self.m:
            
            # Random population
            X = np.random.uniform(low = self.interval[0], high = self.interval[1], size = self.n)
            fitness[i] = self.rastrigin(X)
            B = self.encoding(X)
            population.append(B)

            # Calculating opposite population
            O = self.interval[0] + self.interval[1] - X
            fitness[i+1] = self.rastrigin(O)
            B = self.encoding(O)
            population.append(B)

            # Increases number of individuals
            i += 2
        
        # Selects the m best individuals for initial population
        best_idxs = np.argsort(fitness)[:self.m]
        fitness = fitness[best_idxs]
        population = np.array(population).reshape(((2*self.m),(self.n*self.l)))
        population = population[best_idxs]

        return population, fitness


    def rastrigin(self, X, A = 10):
    
        """ Rastrigin function. """
        
        Y = A*self.n + sum([np.square(x) - A*np.cos(2*np.pi*x) for x in X])
        
        return Y

    def variable_to_variable(self, parent_a, parent_b):
        
        """ One point per variable crossover. """
        
        assert (len(parent_a) == len(parent_b)), "Parents with different lengths"

        prob = float(np.random.uniform(low = 0, high = 1))

        if prob < self.cross_prob:        
            
            # List of crossover points
            points = list()
            
            # Offspring
            offspring_a = np.array(list())
            offspring_b = np.array(list())
            
            # For each variable b in B
            for i in range(self.n):
                
                # Crossover point
                point = int(np.random.randint(low = ((self.l*i)+1), high = (self.l*(i+1)-1), size = 1))
                points.append(point)
                
                # First offspring
                head = parent_a[(self.l*i):point]
                tail = parent_b[point:(self.l*(i+1))]
                offspring_a = np.concatenate([offspring_a, head, tail], axis = 0)
                
                # Second offspring
                head = parent_b[(self.l*i):point]
                tail = parent_a[point:(self.l*(i+1))]
                offspring_b = np.concatenate([offspring_b, head, tail], axis = 0)
                
        else:
            
            # Offspring will be a copy of their parents
            offspring_a = parent_a.copy()
            offspring_b = parent_b.copy()
                        
        return offspring_a, offspring_b

    def mutation(self, parent):
        
        """ Bit-flip mutation. Changes each gene (0 to 1 or 1 to 0) with a probability mut_prob. """
        
        # Offspring
        offspring = parent.copy()

        for i in range(self.n*self.l):
            
            prob = float(np.random.uniform(low = 0, high = 1))
            
            if prob < self.mut_prob:
                offspring[i] = int(not(parent[i]))
                
        return offspring

    def roulette_wheel(self, population, fitness):
        
        """ Fitness proportionate selection. """
        
        # Segment sizes
        a = np.ones((self.m+1,))
        
        # Building the Roulette Wheel
        for i in range(self.m):
            # Minimization problem
            a[i+1] = 1 - sum(fitness[:(i+1)])/sum(fitness)

        # Indexes of selected parents
        idxs = np.zeros((2,))
        
        for i in range(2):
            r = float(np.random.uniform(low = 0, high = 1))
            idxs[i] = np.argmin(r < a) - 1
        
        # Selected parents
        parent_a, parent_b = population[idxs.astype(int)]
            
        return parent_a, parent_b

    def tournament_selection(self, population, fitness):
        
        """ Tournament selection. Deterministic and without replacement. """
        
        # Indexes of selected parents
        idxs = np.zeros((2,))
        
        for i in range(2):
            
            # Candidate solutions
            candidates = np.random.choice(range(self.m), size = self.k, replace = False)
            candidate_fitness = fitness[candidates]        

            # Selects the best individual in the tournament
            idxs[i] = np.argmin(candidate_fitness)
        
        # Selected parents
        parent_a, parent_b = population[idxs.astype(int)]
            
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

        """ Minimizes the Rastrigin function. """

        # Initializes population with opposition-based candidate solutions
        population, fitness = self.opposition_based_population()
        # population, fitness = self.random_initial_population()

        # Progress bar
        pbar = tqdm(total = self.max_eval, desc = 'Function evaluations')

        # Until the maximum number of function evaluations is reached or the global optimal solution is found
        while self.best_fitness > 0 and self.n_eval < self.max_eval:

            # Selects parents
            prob = float(np.random.uniform(low = 0, high = 1))
            
            if prob > 0.5:
                parent_a, parent_b = self.roulette_wheel(population, fitness)
            else:
                parent_a, parent_b = self.tournament_selection(population, fitness)

            # lmbda = int(np.round(self.m/2)) if np.round(self.m/2) % 2 == 0 else int(np.round(self.m/2) + 1)

            # Recombines pairs of parents
            offspring_a, offspring_b = self.variable_to_variable(parent_a, parent_b)

            # Mutates the resulting offspring
            offspring_a = self.mutation(offspring_a)
            offspring_b = self.mutation(offspring_b)

            # Evaluates new candidates
            fitness_a = self.rastrigin(self.decoding(offspring_a))
            fitness_b = self.rastrigin(self.decoding(offspring_b))    

            # Adds new individuals to the population
            population = np.vstack([population, [offspring_a, offspring_b]])
            fitness = np.append(fitness, [fitness_a, fitness_b])        
            
            # Increases number of function evaluations
            self.n_eval = self.n_eval + 2
            pbar.update(2)

            # Selects individuals for the next generation
            population, fitness = self.survivor_selection(population, fitness)

            # Updates new best solution
            self.best_fitness = np.min(fitness)
            self.best_solution = population[np.argmin(fitness)]

            # if self.n_eval % 100 == 0:
            #     plt.scatter(self.n_eval, self.best_fitness, c = 'black', s = 1)
            #     plt.pause(0.00001)

        pbar.close()

if __name__ == '__main__':

    GA = GeneticAlgorithm(n = 10, m = 20, l = 20, max_eval = 10000, mut_prob = 0.025, cross_prob = 0.75, k = 20)
    GA.solve()
    print('Best solution:', GA.best_solution)
    print('Best solution:', GA.decoding(GA.best_solution))
    print('Best fitness:', GA.best_fitness)