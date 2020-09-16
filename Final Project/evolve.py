#pip install numpy
import numpy as np
import subprocess

class Optimizer():

    """ Genetic algorithm to optimize hyperparameters of the YOLO object detector """

    def __init__(self, m, n = 4):

        """
            m: Number of individuals in the population
            n: Number of hyperparameters
        """

        self.m = m
        self.n = n
        self.best_hyperparams = np.zeros((1, self.n))
        self.best_fitness = 0

    def evaluate(self):

        # Running evaluation algorithm and saving to temporary file
        f = open("temp.txt", "w+")
        subprocess.call('./darknet detector map dfire.data dfire.cfg weights/dfire_best.weights', shell = True, stdout = temp)

        # Reading temporary file to extract map metric
        f.seek(0,0)
        for line in f.readlines():
            if 'mAP@0.50' in line:
                fitness = float(line.split('= ')[-1].split(',')[0])
        f.close()

        return fitness

    def train(self):

        # Train model
        subprocess.call('./darknet detector train dfire.data dfire.cfg yolov4.conv.137 -dont_show -map', shell = True)

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
            elif 'iou_thresh' in line:
                lines[i] = 'iou_thresh=' + str(hyperparams['iou_thresh']) + '\n'
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
            momentum = np.random.uniform(low = 0.68, high = 0.98)
            decay = np.random.uniform(low = 0, high = 0.0005)
            lr = np.random.uniform(low = 1e-5, high = 1e-2)
            iou_t = np.random.uniform(low = 0.1, high = 0.7)
            
            # Updating hyperparameters in the configuration file
            hyperparams = {'momentum': momentum, 'decay': decay, 'learning_rate': lr, 'iou_thresh': iou_t}
            self.update(hyperparams)
            # Training the current model
            self.train()
            # Evaluating the current model
            fitness[i] = self.evaluate()

            # Adding individual to the population
            population.append(np.stack((momentum, decay, lr, iou_t)))

        population = np.array(population)

        # Saving the best solution
        self.best_hyperparams = population[np.argmax(fitness)]
        self.best_fitness = np.argmax(fitness)

        print(population, '\n')
        print(fitness)

        return population, fitness

if __name__ == '__main__':

    optimizer = Optimizer(m = 3)
    optimizer.random_initial_population()
    print('Best hyperparameters:', optimizer.best_hyperparams)
    print('Best fitness:', optimizer.best_fitness)