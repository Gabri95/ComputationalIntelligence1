
import numpy as np
import subprocess
import glob
import os.path
import argparse
import sys


class Trainer():
    
    def __init__(self, N, file='../best.params', I=29, O=4, H=15, init_file=None):
        
        #population size
        self.N = N
        
        self.I = I
        self.O = O
        self.H = H
        
        #length of the genomes
        self.gen_size = (H + O) * (I + H + O)
        
        #file where to store the best solution found
        self.file = file

        # record of the best solution
        self.best_performance = 0.0

        # record of the best solution
        self.worst_performance = 0.0
        
        #current population
        self.population = []
        
        
        if init_file is None:
        
            for i in range(N):
                #append the genotype and its fitness
                self.population.append([randomSparseVector(self.gen_size), 0.0])
        else:
            file = open(init_file, 'r')

            # read the first line
            header = file.readline()
    
            # the first line contains the number of input neurons, output neurons and hidden neurons, separated by a comma
            self.I, self.O, self.H = (int(n) for n in header[1:].split(','))
            
            # Matrix containing the connections from each neuron (input, hidden or output) to each hidden or output neuron.
            # As a result, the size has to be (H+O)*(I+H+O).
            # Moreover, the neurons have to be in the following order: input, output, hidden.
            W = np.genfromtxt(init_file, skip_header=1)
    
            assert (self.H + self.O) * (self.I + self.H + self.O) == len(W), "Error! Shape of the parameters not valid!"

            self.population.append([W, 0.0])
            
            for i in range(N/2):
                #append the genotype and its fitness
                self.population.append([self.mutation(W, strength=3), 0.0])
            
            while len(self.population) < N:
                self.population.append([randomSparseVector(self.gen_size), 0.0])
            

    def evaluateGenome(self, genome):
        
        save_genome('../rnd.param', genome, self.I, self.O, self.H)

        subprocess.call(['time', 'python', '../torcs-server/torcs_tournament.py', '../config/quickrace.yml'])
        
        result_file = open('../model_results/results', 'r')
        
        distance, time = [float(x) for x in result_file.readline().split(',')]
        
        return distance + 0.001*distance/(time+1)
        
        # results_file = max(glob.iglob('../torcs-client/output/scr_server 1 - results-*.xml', key=os.path.getmtime))
        #
        # tree = etree.parse(results_file)
        # root = tree.getroot()
        #
        # time = root.xpath('//attnum[@name="time"]/@val')[0]
        # laps = root.xpath('//attnum[@name="laps"]/@val')[0]
        #
        # return float(time)
        
        
    
    def evaluatePopulation(self):
        
        # new_best = -1
        # new_worst = -1
        
        for i, p in enumerate(self.population):
            #In order to not re-evaluate again the genomes survived from the previous generation
            if self.population[i][1] <= 0.0:
                print('\t\tEVALUATING {}'.format(i))
                self.population[i][1] = self.evaluateGenome(self.population[i][0])
                print('\t\tEVALUATED {}: {}'.format(i, self.population[i][1]))
                
                if self.population[i][1] > self.best_performance:
                    self.best_performance = self.population[i][1]
                    #new_best = i
                    save_genome(self.file, self.population[i][0], self.I, self.O, self.H)
                
                if self.population[i][1] < self.worst_performance:
                    self.worst_performance = self.population[i][1]
                    #new_worst = i
                    save_genome('../worst.params', self.population[i][0], self.I, self.O, self.H)
                
        # if new_best >= 0:
        #     save_genome(self.file, self.population[new_best][0], self.I, self.O, self.H)
        #
        # if new_worst >= 0:
        #     save_genome('../worst.params', self.population[new_worst][0], self.I, self.O, self.H)
    
    
    def crossover(self, g1, g2):
    
        p1 = np.random.randint(0, len(g1))
        p2 = np.random.randint(0, len(g1)-1)
        
        if p2 >= p1:
            p2 += 1
        
        pivot1 = min(p1, p2)
        pivot2 = max(p1, p2)
        
        c1 = np.concatenate((g1[:pivot1], g2[pivot1:pivot2], g1[pivot2:]))
        c2 = np.concatenate((g2[:pivot1], g1[pivot1:pivot2], g2[pivot2:]))
        
        return c1, c2
    
    def mutation(self, g, strength=1):
        if np.random.rand() < 0.75:
            return g + np.random.normal(0, strength, len(g)) * np.random.binomial(1, 0.1, len(g))
        else:
            m = np.random.binomial(1, 0.05, len(g))
            return (1-m)*g + m*np.random.normal(0, strength, len(g))
    
    def epoch(self):
        
        self.evaluatePopulation()

        self.population.sort(key=lambda x : -x[1])
        
        
        
        size = len(self.population)
        
        #drop all genotypes with non positive fitness
        self.population[:] = [p for p in self.population if p[1] > 0]
        
        #dropp worst genotypes in order to preserve only 1/5 of the original population
        del self.population[max(size//5, 1):]
        
        
        #in case we dropped to many genotypes
        while len(self.population) < min(4, size):
            self.population.append([randomSparseVector(self.gen_size), 0.0])

        survived = [p for p, f in self.population]
        
        #fist genotypes are more likely to be choosen
        survived_weights = [2**(-0.3*i) for i in range(len(survived))]
        
        #repopulation
        while len(self.population) < size:
            
            #with a small probability one of the survived genomes mutates
            if np.random.random() < 0.2 + 0.3*(size - len(survived))/size:
                c = self.mutation(survived[sample(survived_weights)]) #np.random.randint(0, len(survived))])
                self.population.append([c, 0.0])
            
            else:
                i1 = sample(survived_weights) # np.random.randint(0, len(survived))
                i2 = sample([w for i, w in enumerate(survived_weights) if i!=i1]) #np.random.randint(0, len(survived))
                
                if i2 >= i1:
                    i2 += 1
                
                c1, c2 = self.crossover(survived[i1], survived[i2])
                
                self.population.append([c1, 0.0])
                
                if len(self.population) < size:
                    self.population.append([c2, 0.0])
        
        
        
        
    
    def train(self, E):
    
        for e in range(E):
            print(' - - - - - - - - - - - - - - -  E P O C H  {}  - - - - - - - - - - - - - - -'.format(e))
            
            self.epoch()
            
            print('\tBest performance so far: {}'.format(self.best_performance))
            
    
    
        
        
def randomSparseVector(n):
    v = np.random.normal(0, 1, n)
    
    m = np.random.binomial(1, 0.2, n)
    
    return v*m
    

def save_genome(file, parameters, I, O, H):
    np.savetxt(file, parameters.reshape(-1), header=str(I) + ', ' + str(O) + ', ' + str(H))
    

def sample(P):
    u = sum(P)*np.random.rand()
    p=0
    for i, p_i in enumerate(P):
        p+=p_i
        
        if u <= p:
            return i
    
    return len(P)-1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trainer for the genetic algorithm'
    )
    parser.add_argument(
        '-i',
        '--init_file',
        help='Model parameters to use as initializzation.',
        type=str
    )

    args, _ = parser.parse_known_args()
    
    trainer = Trainer(30, init_file=args.init_file)
    
    trainer.train(30)