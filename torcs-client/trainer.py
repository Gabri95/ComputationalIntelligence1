
import numpy as np
import subprocess

class Trainer():
    
    def __init__(self, N, file, I=77, O=1, H=30):
        
        #population size
        self.N = N
        
        self.I = I
        self.O = O
        self.H = H
        
        #length of the genomes
        self.gen_size = (H + O) * (I + H + O)
        
        #file where to store the best solution found
        self.file = file
        
        #record of the best solution
        self.best_performance = 0.0
        
        #current population
        self.population = []
        
        
        
        
        for i in range(N):
            #append the genotype and its fitness
            self.population.append((np.random.normal(0, 1, (self.gen_size,)), 0.0))
    
    
    
    
    def evaluateGenome(self, genome):
        
        save_genome('../rnd.params', genome, self.I, self.O, self.H)

        subprocess.call(['python', '../torcs-server/torcs_tournament.py', '../config/quickrace.yml'])
        
        ratings = subprocess.check_output(['tail', '-1', '../ratings.csv'])
        
        return float(ratings.split(',')[1])
        
    
    def evaluatePopulation(self):
        
        new_best = -1
        
        for i, p in enumerate(self.population):
            #In order to not re-evaluate again the genomes survived from the previous generation
            if self.population[i][1] <= 0.0:
                
                self.population[i][1] = self.evaluateGenome(self.population[i][0])
                
                if self.population[i][1] > self.best_performance:
                    self.best_performance = self.population[i][1]
                    new_best = i
                
        if new_best >= 0:
            save_genome('../best.params', self.population[new_best][0], self.I, self.O, self.H)
    
    
    def crossover(self, g1, g2):
    
        pivot = np.random.randint(0, len(g1))

        c1 = np.concatenate((g1[:pivot], g2[pivot:]))
        c2 = np.concatenate((g2[:pivot], g1[pivot:]))
        
        return c1, c2
    
    def mutation(self, g):
        return g + np.random.normal(0, 0.1, (len(g),))
    
    def epoch(self):
        
        self.evaluatePopulation()

        self.population.sort(key=lambda x : x[1])
        
        
        survived = [p for p, f in self.population[:len(self.population)//5]]
        
        size = len(self.population)

        del self.population[len(self.population)//5:]
        
        #self.population = self.population[:len(self.population)//5]
        
        while len(self.population) < size:
            
            #with a small probability one of the survived genome mutates
            if np.random.random() < 0.2:
                c = self.mutation(survived[np.random.randint(0, len(survived))])
            else:
                c1, c2 = self.crossover(survived[np.random.randint(0, len(survived))], survived[np.random.randint(0, len(survived))])
                
                self.population.append((c1, 0.0))
                
                if len(self.population) < size:
                    self.population.append((c2, 0.0))
        
        
        
        
    
    def train(self, N):
    
        for e in range(N):
            print('---- Starting epoch {} ----'.format(e))
            
            self.epoch()
            
            print('\tBest performance so far: {}'.format(self.best_performance))
            
    
    
        
        
        

def save_genome(file, parameters, I, O, H):
    np.savetxt(file, parameters.reshape(-1), header=str(I) + ', ' + str(O) + ', ' + str(H))
    