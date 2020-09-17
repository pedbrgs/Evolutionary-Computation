import subprocess
#pip install numpy
import numpy as np
# pip install tqdm
from tqdm import tqdm

class Optimizer():

    """ Genetic algorithm to optimize hyperparameters of the Tiny-YOLO object detector """

    def __init__(self, m, k, max_gen, n = 4, mut_prob = 0.25, cross_prob = 0.8):

        """
            m: Number of individuals in the population
            k: Sample size in parent selection
            max_gen: Maximum number of generations
            n: Number of hyperparameters
            mut_prob: Mutation probability
            cross_prob: Crossover probability
        """

        self.m = m
        self.n = n
        self.k = k
        self.best_fitness = 0
        self.max_gen = max_gen
        self.mut_prob = mut_prob
        self.cross_prob = cross_prob
        self.best_hyperparams = np.zeros((1, self.n))

    def evaluate(self):

        # Running evaluation algorithm and saving to temporary file
        f = open("temp.txt", "w+")
        subprocess.call('./darknet detector map dfire.data dfire.cfg weights/dfire_final.weights', shell = True, stdout = f)

        # Reading temporary file to extract map metric
        f.seek(0,0)
        for line in f.readlines():
            if 'mAP@0.50' in line:
                fitness = float(line.split('= ')[-1].split(',')[0])
        f.close()

        return fitness

    def train(self):

        # Train model
        subprocess.call('./darknet detector train dfire.data dfire.cfg yolov4-tiny.conv.29 -dont_show -map', shell = True)

    def update(self, hyperparams):

        cfg = open("dfire.cfg", "r")
        lines = cfg.readlines()

        # Updating hyperparameters in the model
        for i, line in enumerate(lines):
            if 'momentum' in line:
                lines[i] = 'momentum=' + str(hyperparams['momentum']) + '\n'
            elif 'decay' in line:
                lines[i] = 'decay=' + str(hyperparams['decay']) + '\n'
            elif 'learning_rate' in line:
                lines[i] = 'learning_rate=' + str(hyperparams['learning_rate']) + '\n'
            elif 'ignore_thresh' in line:
                lines[i] = 'ignore_thresh=' + str(hyperparams['ignore_thresh']) + '\n'
            else:
                pass
        
        cfg = open("dfire.cfg", "w")
        # Writing new hyperparameters in the configuration file
        cfg.writelines(lines)
        cfg.close()

    def random_initial_population(self):

        # Empty population
        population = list()
        fitness = np.zeros((self.m,))

        # Generating initial population
        for i in range(self.m):
          
            # Generating hyperparameter values randomly
            if i != 0:
                momentum = np.random.uniform(low = 0.68, high = 0.98)
                decay = np.random.uniform(low = 0, high = 5e-4)
                lr = np.random.uniform(low = 1e-4, high = 1e-2)
                thresh = np.random.uniform(low = 0.4, high = 0.8)       
            # Default hyperparameters     
            else:
                momentum, decay, lr, thresh = 0.9, 0.0005, 0.001, 0.7

            hyperparams = {'momentum': momentum, 'decay': decay, 'learning_rate': lr, 'ignore_thresh': thresh}

            # Updating hyperparameters in the configuration file
            self.update(hyperparams)
            
            # Training the current model
            self.train()
            
            # Evaluating the current model
            fitness[i] = self.evaluate()

            # Adding individual to the population
            population.append(np.stack((momentum, decay, lr, thresh)))

        population = np.array(population)

        # Saving the best solution
        self.best_hyperparams = population[np.argmax(fitness)]
        self.best_fitness = np.max(fitness)

        return population, fitness

    def crossover(self, parent_a, parent_b):

        """ Single point crossover. """

        prob = float(np.random.uniform(low = 0, high = 1))

        if prob < self.cross_prob:

            # Offspring
            offspring_a = np.array(list())
            offspring_b = np.array(list())

            # Crossover point
            point = int(np.random.randint(low = 1, high = self.n-1, size = 1))

            # First offspring
            head = parent_a[:point]
            tail = parent_b[point:]
            offspring_a = np.concatenate([offspring_a, head, tail], axis = 0)

            # Second offspring
            head = parent_b[:point]
            tail = parent_a[point:]
            offspring_b = np.concatenate([offspring_b, head, tail], axis = 0)

        else:

            # Offspring will be a copy of their parents
            offspring_a, offspring_b = parent_a.copy(), parent_b.copy()

        return offspring_a, offspring_b
    
    def mutation(self, parent):

        """ Uniform mutation. """

        # Offspring
        offspring = parent.copy()

        # Probability of each hyperparameter mutating
        probs = np.random.uniform(low = 0, high = 1, size = self.n)
        mask = (probs < self.mut_prob) * 1

        # Possible new hyperparameters
        momentum = np.random.uniform(low = 0.68, high = 0.98)
        decay = np.random.uniform(low = 0, high = 5e-4)
        lr = np.random.uniform(low = 1e-4, high = 1e-2)
        thresh = np.random.uniform(low = 0.4, high = 0.8)

        # Mutation
        mutation = mask * np.array([momentum, decay, lr, thresh])
        offspring[np.where(mutation != 0)] = mutation[np.where(mutation != 0)]

        return offspring

    def parent_selection(self, population, fitness):
        
        """ Selects two parents from a sample of the population. """

        # Candidate solutions
        idxs = np.random.choice(range(self.m), size = self.k, replace = False)
        candidate_solutions = population[idxs]
        candidate_fitness = fitness[idxs]        
        
        # Selects the two best individuals for crossing
        best_idxs = np.argsort(candidate_fitness)[-2:]
        parent_a = candidate_solutions[best_idxs[0]]
        parent_b = candidate_solutions[best_idxs[1]]
        
        return parent_a, parent_b

    def survivor_selection(self, population, fitness):

        """ Selection of individuals who will remain in the population. """
        
        # Selects the two worst individuals in the population
        idxs = np.argsort(fitness)[:2]
        
        # Eliminates the two worst individuals in the population
        population = np.delete(population, idxs, axis = 0)
        fitness = np.delete(fitness, idxs)
        
        return population, fitness

    def evolve(self):

        # Initializes population with random hyperparameters
        population, fitness = self.random_initial_population()

        # Log
        log = open("log.txt", "w+")
        log.write('momentum,decay,learning_rate,ignore_thresh,mAP\n')

        # Logging best hyperparameters and best fitness
        log.write(str(np.round(self.best_hyperparams[0], 6)) + ',' + 
                  str(np.round(self.best_hyperparams[1], 8)) + ',' + 
                  str(np.round(self.best_hyperparams[2], 8)) + ',' + 
                  str(np.round(self.best_hyperparams[3], 6)) + ',' + 
                  str(self.best_fitness) + '\n')

        # Progress bar
        pbar = tqdm(total = self.max_gen, desc = 'Generations')

        # Until the maximum number of generations is reached
        for n_gen in range(self.max_gen):

            # Selects parents
            parent_a, parent_b = self.parent_selection(population, fitness)

            # Recombines pairs of parents
            offspring_a, offspring_b = self.crossover(parent_a, parent_b)
            
            # Mutates the resulting offspring
            offspring_a = self.mutation(offspring_a)
            offspring_b = self.mutation(offspring_b)

            # Evaluates first offspring
            hyperparams = {'momentum': offspring_a[0], 'decay': offspring_a[1], 'learning_rate': offspring_a[2], 'ignore_thresh': offspring_a[3]}
            self.update(hyperparams)          
            self.train()
            fitness_a = self.evaluate()
            
            # Evaluates second offspring
            hyperparams = {'momentum': offspring_b[0], 'decay': offspring_b[1], 'learning_rate': offspring_b[2], 'ignore_thresh': offspring_b[3]}
            self.update(hyperparams)          
            self.train()
            fitness_b = self.evaluate()

            # Adds new individuals to the population
            population = np.vstack([population, [offspring_a, offspring_b]])
            fitness = np.append(fitness, [fitness_a, fitness_b])            

            # Selects individuals for the next generation
            population, fitness = self.survivor_selection(population, fitness)

            # Updates new best hyperparameters
            self.best_hyperparams = population[np.argmax(fitness)]
            self.best_fitness = np.max(fitness)

            # Logging best hyperparameters and best fitness
            log.write(str(np.round(self.best_hyperparams[0], 6)) + ',' + 
                      str(np.round(self.best_hyperparams[1], 8)) + ',' + 
                      str(np.round(self.best_hyperparams[2], 8)) + ',' + 
                      str(np.round(self.best_hyperparams[3], 6)) + ',' + 
                      str(self.best_fitness) + '\n')

            # Increases number of generations
            pbar.update(1)       

        log.close()
        pbar.close()

if __name__ == '__main__':

    optimizer = Optimizer(m = 3, k = 3, max_gen = 2, mut_prob = 0.25, cross_prob = 0.8)
    optimizer.evolve()
    print('Best hyperparameters:', optimizer.best_hyperparams)
    print('Best fitness:', optimizer.best_fitness)